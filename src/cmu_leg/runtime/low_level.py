"""
Low-level process body. Runs as a child process.

Owns the EtherCAT master and the ProprioceptiveEstimator. Each tick: reads one
EtherCAT frame -> proprioceptive physical state -> publishes the medulla subset
to the estimator (B1); reads the high-level command (B3) through a staleness
watchdog and computes the low-level drive command via LowLevelControl.

Actuation (`ec.write`) is DEFERRED for safety: the drive command is computed and
logged, but not yet written to the motors -- see the TODO(bench) in the loop.

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
from cmu_leg.estimation.proprioceptive_estimator import ProprioceptiveEstimator
from cmu_leg.control.low_level_control import LowLevelControl
from cmu_leg.types.commands import TorqueDesired, DriveCommand
from cmu_leg.runtime.buffer import MedullaStateBuffer, CommandBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.utility.recorder import Recorder

MAX_STALE_TICKS = 5   # B3 watchdog: no fresh command for this many ticks -> safe zero


@dataclass
class LoopStats:  # <1>
    """Per-tick timing trace for the online idle-margin / loop-period plots.
    Recorder logs dataclasses, so the two scalars need this wrapper."""
    idle: float
    dt: float


def run(dt, tmax, buffer, cmd_buf, start_barrier, stop_event, start_time,
        init_evnt, run_dir, is_verbose=False):
    """Low-level child process entry point (the Process target).

    Args:
        dt (float): Nominal loop period (s) -- steps the sleep-to-clock ONLY.
            Do NOT feed this to the estimator; it needs the MEASURED dt.
        tmax (float): Max elapsed time (s); fixes the integer cycle count.
        buffer (MedullaStateBuffer): B1, low-level -> estimator. Sole writer.
        cmd_buf (CommandBuffer): B3, high-level -> low-level. Sole reader, with a
            staleness watchdog (see <7>).
        start_barrier (multiprocessing.Barrier): Trips once every process reaches
            it; its action stamps the shared start_time. Time-syncs the loops.
        stop_event (multiprocessing.Event): Set by any process (or the guard) to
            request graceful shutdown everywhere.
        start_time (multiprocessing.Value): Shared 'd'; the synced perf_counter
            origin. Read AFTER start_barrier.wait().
        init_evnt (multiprocessing.Event): Set once hardware is up, so the parent
            gates the estimator spawn on it (hgeyer's init_evnt). <2>
        run_dir (str | Path): Log directory, created ONCE by the parent so all
            children's CSVs land together and can be joined on `t`. <3>
        is_verbose (bool): Verbose prints.
    """
    pname = current_process().name

    # completion_stage gates teardown so it only touches what actually came up.
    completion_stage = 0

    def terminate_gracefully():
        gc.enable()  # undo the RT gc freeze unconditionally
        if completion_stage >= 1:
            for lg in (state_log, frame_log, drive_log, timing_log):
                lg.save(run_dir)
            ec.close()
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
        state_log = Recorder("state")     # ProsthesisState each tick
        frame_log = Recorder("frame")     # raw EtherCAT frame each tick
        drive_log = Recorder("drive")     # DriveCommand the loop WOULD send
        timing_log = Recorder("idle")     # LoopStats (idle margin + measured dt)

        completion_stage = 1
        init_evnt.set()   # hardware up -> parent may now spawn the estimator

        # --- RT gc hygiene (mirrors main.py) ---------------------------------
        gc.collect()
        gc.freeze()
        gc.disable()

        # --- sync start (Barrier(3): low-level, estimator, high-level) -------
        if start_barrier.wait() == 0:                 # <4>
            with start_time.get_lock():
                start_time.value = perf_counter()
        start_barrier.wait()
        t0 = start_time.value

        # --- soft-realtime loop (integer cycle count, like devcom) -----------
        clock = t0
        t_prev = t0 - dt          # so the first measured dt ~= dt
        last_cmd_seq = 0          # B3 watchdog state
        stale_ticks = 0
        cycle = 0
        while cycle < num_cycles:  # <5>
            now = perf_counter()
            t = now - t0          # published stamp, relative to synced start

            # TWO dt's, deliberately (see main.py): nominal `dt` steps the clock;
            # measured `dt_meas` drives every BackwardDifference in the estimator.
            dt_meas = now - t_prev
            t_prev = now

            frame = ec.read()
            state = proprio.update(frame, dt_meas)

            # publish the medulla-sourced subset to the estimator (B1)
            buffer.publish(
                state.knee_angle, state.ankle_angle,
                state.knee_velocity, state.ankle_velocity,
                state.toe_grf, state.heel_grf, t,
            )

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
            # nonzero command only when standing on a FRESH command; else zeros
            drive_cmd = (llc.to_drive_command(llo)
                         if (fresh and state.standing) else DriveCommand())

            # TODO(bench): actuation deferred for safety. To drive the motors,
            # switch to the main.py exchange pattern -- read ONCE before the loop
            # and `frame = ec.write(drive_cmd)` here (one PDO exchange per tick) --
            # and manage ec.enable. Until then we stay read-only and just log the
            # command the loop WOULD send.

            if stop_event.is_set():
                return   # guard ensures graceful termination

            # sync to cycle clock; ALWAYS yield the GIL (sleep(0) on overrun)
            clock += dt
            idle = clock - perf_counter()
            state_log.append(t, state)
            frame_log.append(t, frame)
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
