"""
Low-level process body. Runs as a child process.

Owns the EtherCAT master and the ProprioceptiveEstimator. Each tick: derives the
proprioceptive physical state from the last EtherCAT frame -> publishes the full
state to the estimator (B1); reads the high-level command (B3) through a staleness
watchdog, computes the low-level drive command via LowLevelControl, and writes it
back with `ec.write` (one PDO exchange per tick, which also reads the next frame).

BRING-UP SAFETY: `COMMAND_MOTORS` gates whether real torque is sent. With it False
(the default), the drives are still ENERGIZED after WARMUP_S (ec.enable rises,
CiA-402 FSM walks to OPERATION_ENABLED) but the commanded torque is forced to zero
-- this validates the enable/write path with no motion. Flip it True only when the
leg is on a stand and you intend to command torque.

Structured after hgeyer's hardware/devcom.py run(): a run() entry point spawned
as the Process target, hardware initialized AFTER spawn (pysoem master can't be
pickled), a completion_stage-guarded teardown, a start Barrier for time-sync, and
an integer-cycle soft-realtime loop that always yields the GIL.
"""

import gc
from dataclasses import dataclass
from math import floor
from time import perf_counter, sleep
from multiprocessing import current_process

from cmu_leg.hardware.ethercat_io import EtherCATInterface
from cmu_leg.hardware.elmos.state_machine import drive_state
from cmu_leg.estimation.proprioceptive_estimator import ProprioceptiveEstimator
from cmu_leg.control.low_level_control import LowLevelControl
from cmu_leg.types.commands import TorqueDesired, DriveCommand
from cmu_leg.runtime.buffer import StateBuffer, CommandBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.utility.recorder import Recorder

MAX_STALE_TICKS = 5   # B3 watchdog: no fresh command for this many ticks -> safe zero
WARMUP_S = 7.0        # settle time before energizing the drives (ec.enable rises at t >= this)
SHUTDOWN_S = 1.0      # disable-ramp on teardown: exchange zero cmds w/ enable=False this long,
                      # walking the drives to SWITCH_ON_DISABLED before close (mirrors main.py)
COMMAND_MOTORS = True  # BRING-UP GATE: False = enable the drives but always send ZERO torque
                        # (validate the enable/write FSM path). True = send the real drive command.

# Torque trip guard: if the controller commands more motor torque ff [Nm] than this
# on either joint -> de-energize and stop the whole run. Set
# below the controller's own saturation (torque_max = 100 knee / 150 ankle) so it
# catches a runaway before the ceiling. 
# 
# Tune to the torque you expect this session.
KNEE_TORQUE_TRIP = 50.0    # [Nm]
ANKLE_TORQUE_TRIP = 50.0   # [Nm]

ENABLE_GRACE_S = 2.0  # after ec.enable rises, warn if the drives haven't reached
                      # OPERATION_ENABLED within this long (motor power off / STO / fault)


@dataclass
class LoopStats:  # <1>
    """Per-tick timing trace for the online idle-margin / loop-period plots.
    Recorder logs dataclasses, so the two scalars need this wrapper."""
    idle: float
    dt: float


def run(dt, tmax, buffer, cmd_buf, stop_event, start_time,
        init_evnt, run_dir, is_verbose=False):
    """Low-level child process entry point (the Process target).

    Args:
        dt (float): Nominal loop period (s) -- steps the sleep-to-clock ONLY.
            Do NOT feed this to the estimator; it needs the MEASURED dt.
        tmax (float): Max elapsed time (s); fixes the integer cycle count.
        buffer (StateBuffer): B1, low-level -> estimator. Sole writer; carries
            the full proprioceptive ProsthesisState (hip_* still zero here).
        cmd_buf (CommandBuffer): B3, high-level -> low-level. Sole reader, with a
            staleness watchdog (see <7>).
        stop_event (multiprocessing.Event): Set by any process (or the guard) to
            request graceful shutdown everywhere.
        start_time (multiprocessing.Value): Shared 'd'; the synced perf_counter
            origin, STAMPED by the estimator+high-level barrier. Low-level is not
            in that barrier -- it heartbeats the EtherCAT bus and reads this once
            it becomes nonzero (see the start-sync note below).
        init_evnt (multiprocessing.Event): Set once hardware is up, so the parent
            gates the estimator spawn on it (hgeyer's init_evnt). <2>
        run_dir (str | Path): Log directory, created ONCE by the parent so all
            children's CSVs land together and can be joined on `t`. <3>
        is_verbose (bool): Verbose prints.
    """
    pname = current_process().name

    # completion_stage gates teardown so it only touches what actually came up.
    completion_stage = 0
    ec = None                     # bound before the guard so teardown can always see it

    def terminate_gracefully():
        gc.enable()  # undo the RT gc freeze unconditionally
        if completion_stage >= 1:
            for lg in (state_log, frame_log, output_log, drive_log, timing_log):
                lg.save(run_dir)
        # ALWAYS de-energize + reset the bus if the master ever came up -- even on an
        # init-time failure (an emergency during ec.connect() raises BEFORE
        # completion_stage=1). Two layers:
        #  1. If the loop actually ran (completion_stage>=1, so the bus reached OP),
        #     run a disable-ramp: keep exchanging ZERO commands with enable=False so
        #     the Elmos walk the CiA-402 FSM down to SWITCH_ON_DISABLED (torque off,
        #     clean stop) rather than being cut mid-torque. Mirrors main.py's
        #     SHUTDOWN_TIME phase.
        #  2. ALWAYS ec.close() (in finally): drives the bus to INIT -> de-energized
        #     and reset for the next run. This is the backstop even if the ramp is
        #     skipped (init failure) or fails. The old completion_stage gate skipped
        #     close entirely on init failure, which left the motors humming.
        if ec is not None:
            try:
                ec.enable = False
                if completion_stage >= 1:
                    for _ in range(int(SHUTDOWN_S / dt)):
                        ec.write(DriveCommand())
            except Exception as e:
                print(f"{pname}: disable-ramp failed during teardown: {e}")
            finally:
                try:
                    ec.close()
                except Exception as e:
                    print(f"{pname}: EtherCAT close failed during teardown: {e}")
        if is_verbose:
            print(f"{pname}: terminated gracefully.")

    # A SIGTERM/SIGINT/exception sets stop_event and still runs teardown.
    with ExIntGuard(pname, stop_event, terminate_gracefully):
        # --- stage 1: init hardware AFTER spawn ------------------------------
        ec = EtherCATInterface("eth0")
        ec.connect()
        proprio = ProprioceptiveEstimator()
        llc = LowLevelControl()

        # preallocate logs (memory-only append in the loop; disk write at teardown)
        num_cycles = floor(tmax / dt) + 1
        state_log = Recorder("low_prosthesis_state")  # ProsthesisState each tick
        frame_log = Recorder("low_frame")     # raw EtherCAT frame each tick
        output_log = Recorder("low_output")   # LowLevelOutput (llo) -- the control output
        drive_log = Recorder("low_drive")     # DriveCommand the loop WOULD send
        timing_log = Recorder("low_idle")     # LoopStats (idle margin + measured dt)

        completion_stage = 1
        init_evnt.set()   # hardware up -> parent may now spawn the estimator

        # --- RT gc hygiene (mirrors main.py) ---------------------------------
        gc.collect()
        gc.freeze()
        gc.disable()

        # --- start sync: HEARTBEAT, do not block on a barrier ----------------
        # Low-level is deliberately NOT in the start Barrier. Blocking here would
        # leave the EtherCAT bus silent while the estimator inits its IMUs (~4 s),
        # and the Elmo sync-manager watchdog would drop the drives out of OP (they
        # then ignore the output PDO -> stuck in SWITCH_ON_DISABLED). Instead we
        # keep exchanging to hold OP, and pick up the shared origin the estimator+
        # high-level barrier stamps into start_time. Same t0, warm bus.
        while start_time.value == 0.0 and not stop_event.is_set():
            ec.read()   # heartbeat: one PDO exchange -> feeds the drive watchdog
        if stop_event.is_set():
            return
        t0 = start_time.value

        # --- soft-realtime loop (integer cycle count, like devcom) -----------
        clock = t0
        t_prev = t0 - dt          # so the first measured dt ~= dt
        last_cmd_seq = 0          # B3 watchdog state
        stale_ticks = 0
        cycle = 0
        reached_enabled = False   # drives confirmed OPERATION_ENABLED at least once
        enable_since = None       # t when ec.enable first went True
        warned_enable = False     # one-shot "never reached enabled" warning
        # Seed the input frame with one read so the first ec.write derives its
        # control word from a real status word (write, then read-back, per tick).
        frame = ec.read()
        while cycle < num_cycles:  # <5>
            now = perf_counter()
            t = now - t0          # published stamp, relative to synced start

            # TWO dt's, deliberately (see main.py): nominal `dt` steps the clock;
            # measured `dt_meas` drives every BackwardDifference in the estimator.
            dt_meas = now - t_prev
            t_prev = now

            # state is derived from the frame read back by the PREVIOUS exchange
            # (one-cycle sense->act latency, same as main.py's read-once pattern).
            state = proprio.update(frame, dt_meas)

            # publish the full proprioceptive state to the estimator (B1). hip_*
            # is still zero here (no ESKF in low-level); the estimator overlays it.
            buffer.publish(state)

            # --- read the high-level command (B3) through the watchdog -------
            cmd = cmd_buf.latest()
            if cmd is None or cmd.seq == last_cmd_seq:
                stale_ticks += 1
            else:
                stale_ticks = 0
                last_cmd_seq = cmd.seq
            fresh = cmd is not None and stale_ticks < MAX_STALE_TICKS   # <7>

            if fresh:
                state.standing = cmd.standing
                desired = TorqueDesired(cmd.knee_desired, cmd.ankle_desired)
            else:
                state.standing = False           # watchdog tripped -> safe zero
                desired = TorqueDesired()

            llo = llc.compute(state, desired_torque=desired,
                              control_enabled=ec.control_enabled(), dt=dt_meas)
            # Real command only when COMMAND_MOTORS is set AND standing on a FRESH
            # command; else zeros. COMMAND_MOTORS=False forces zeros for the enable-
            # validation run. (The standing gate is a stopgap -- real stance/swing
            # control should command torque regardless of standing.)
            drive_cmd = (llc.to_drive_command(llo)
                         if (COMMAND_MOTORS and fresh and state.standing) else DriveCommand())

            # Torque trip guard: gate on the controller's COMMANDED motor torque ff
            # (deterministic, watches what the controller WOULD drive even while
            # COMMAND_MOTORS forces zeros). An over-limit command means something is
            # wrong -> command zero + disable this tick, then stop the whole run.
            if (abs(llo.knee_motor_torque_ff) > KNEE_TORQUE_TRIP
                    or abs(llo.ankle_motor_torque_ff) > ANKLE_TORQUE_TRIP):
                print(f"[TORQUE TRIP] knee={llo.knee_motor_torque_ff:.2f} "
                      f"ankle={llo.ankle_motor_torque_ff:.2f} Nm -> de-energizing")
                ec.enable = False
                ec.write(DriveCommand())   # zero command + disable this tick
                stop_event.set()           # propagate the stop to estimator + high-level
                return                     # ExIntGuard runs teardown (saves logs, ec.close)

            # Energize after the warmup: ec.write drives the CiA-402 FSM toward
            # OPERATION_ENABLED whenever ec.enable is True. Set enable BEFORE the
            # write so this tick's control word reflects it.
            ec.enable = t >= WARMUP_S
            frame = ec.write(drive_cmd)   # one PDO exchange/tick; returns fresh inputs

            # --- drive-enable feedback (parity with main.py's reached_enabled) ---
            # Report the first time the drives reach OPERATION_ENABLED, and warn
            # once if they never do within ENABLE_GRACE_S of commanding enable --
            # showing the CiA-402 state each drive is stuck in (motor power / STO).
            if ec.enable and enable_since is None:
                enable_since = t
            if not reached_enabled and ec.control_enabled():
                reached_enabled = True
                print(f"[{t:6.3f}s] drives OPERATION_ENABLED")
            elif (enable_since is not None and not reached_enabled and not warned_enable
                  and t - enable_since > ENABLE_GRACE_S):
                warned_enable = True
                knee_st = drive_state(frame.knee_status_word).name
                ankle_st = drive_state(frame.ankle_status_word).name
                print(f"[{t:6.3f}s] WARNING: drives did NOT reach OPERATION_ENABLED "
                      f"{ENABLE_GRACE_S:.0f}s after enable (knee={knee_st}, ankle={ankle_st}). "
                      f"Check motor power / STO / faults.")

            if stop_event.is_set():
                return   # guard ensures graceful termination

            # sync to cycle clock; ALWAYS yield the GIL (sleep(0) on overrun)
            clock += dt
            idle = clock - perf_counter()
            state_log.append(t, state)
            frame_log.append(t, frame)
            output_log.append(t, llo)
            drive_log.append(t, drive_cmd)
            timing_log.append(t, LoopStats(idle, dt_meas))
            sleep(idle) if idle >= 0.0 else sleep(0.0)   # <6>

            cycle += 1
    # (ExIntGuard runs terminate_gracefully on exit -- no finally needed)

# --------------------------------------------------------------------
# <1>: Recorder flattens a dataclass into CSV columns; a bare float can't be
# logged (dataclasses.fields() rejects it), so timing goes through LoopStats.
#
# <2>: The parent spawns this process, blocks on init_evnt (hgeyer's
# hw_init_ok), and only then spawns the estimator -- so the consumer never comes
# up before the EtherCAT hardware is confirmed live.
#
# <3>: Each child logs its OWN trace (logging stays off every hot path and out
# of IPC). The parent makes one run_dir and passes it to all so the CSVs share a
# directory and timeline.
#
# <4>: Two-phase start sync. Every barriered process reaches here; whoever gets
# index 0 stamps perf_counter() into the shared start_time, and the second wait
# guarantees the others read it only after. (Avoids a Barrier action, which is
# awkward to pickle to spawned children.)
#
# <5>: Integer cycle counting (not `while clock < stop`) avoids rounding drift
# and makes the log length exactly num_cycles.
#
# <6>: sleep(0.0) even on overrun forces a GIL yield so pysoem's background
# threads get a turn.
#
# <7>: B3 staleness watchdog. If high-level hangs/dies its seq stops advancing;
# after MAX_STALE_TICKS with no fresh command we force a safe zero rather than
# holding the last (possibly nonzero) torque. Safety-critical.
