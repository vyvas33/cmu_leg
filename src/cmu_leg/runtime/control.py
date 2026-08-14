"""
Control process body (P2). Runs as a child process.

Owns the two USB IMUs, the ESKF, AND the high-level controller -- the estimator
and high-level stages, MERGED into one process (co-located, hgeyer-aligned). Each
tick: reads the IMUs, pulls the latest full ProsthesisState off the shared B1
StateBuffer, runs the ESKF to overlay hip_*, decides the mode (standing/walking),
runs the active behavioral policy, and publishes the desired joint torques + mode
flag straight to B3 (CommandBuffer, down to the low-level process).

WHY MERGED: running the ESKF in its own process put an extra IPC hop (and a
500 Hz vs 1 kHz rate mismatch) between the sensor and the controller, so the
standing impedance's velocity-damping term acted on ~3-4 ms-stale velocity and
drove an 11 Hz limit cycle. Co-locating estimation + control halves that transport
delay (2 hops -> 1). See the multiprocess design notes. The fast, delay-sensitive
velocity still lives at the sensor (low-level, 1 kHz); if standing still oscillates,
the damping term moves DOWN into the low-level -- it can't be fixed up here.

Symmetric to runtime/low_level.py: a run() Process target, hardware initialized
AFTER spawn (USB handles can't be pickled), a completion_stage-guarded teardown,
and an integer-cycle soft-realtime loop. There is no start Barrier anymore: this
is the only child that stamps start_time (low-level heartbeats the bus until it
becomes nonzero), so it stamps directly after IMU init.
"""

import gc
from dataclasses import dataclass
from math import floor
from time import perf_counter, sleep
from multiprocessing import current_process

from cmu_leg.hardware.imu_io import IMUInterface
from cmu_leg.estimation.eskf.eskf import ESKF
from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.control.utils import BackwardDifference
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import Mode
from cmu_leg.runtime.operator import TerminalOperator, TimedOperator
from cmu_leg.runtime.buffer import StateBuffer, CommandBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.utility.recorder import Recorder
from cmu_leg.utility.rtutils import pin_to_core
import cmu_leg.config.params as params

# Standing is the resting request (see runtime.operator); the drives gate it
# in run()). TODO: replace the time schedule with the real mode FSM / supervisor.


@dataclass
class LoopStats:
    """Per-tick timing trace for the online idle-margin / loop-period plots.
    Recorder logs dataclasses, so the two scalars need this wrapper."""
    idle: float
    dt: float


def run(dt, tmax, buffer, cmd_buf, stop_event, start_time,
        run_dir, is_verbose=False, core=None, rt_priority=None):
    """Control child process entry point (the Process target).

    Args:
        dt (float): Nominal loop period (s) -- steps the sleep-to-clock ONLY.
            The controller runs at THIS rate (~500 Hz); the ESKF integration and
            every BackwardDifference use the MEASURED dt instead.
        tmax (float): Max elapsed time (s); fixes the integer cycle count.
        buffer (StateBuffer): B1, low-level -> control. Sole reader; carries the
            full proprioceptive ProsthesisState (hip_* zero until we overlay it).
        cmd_buf (CommandBuffer): B3, control -> low-level. Sole writer; desired
            joint torques + the standing mode flag.
        stop_event (multiprocessing.Event): Graceful-shutdown request.
        start_time (multiprocessing.Value): Shared 'd' synced perf_counter origin.
            Control is the sole stamper (no barrier); low-level heartbeats the bus
            until this becomes nonzero, then both run from the common origin.
        run_dir (str | Path): Log directory, created ONCE by the parent so both
            children's CSVs land together and can be joined on `t`.
        is_verbose (bool): Verbose prints.

    No init_evnt here: nothing downstream is gated on control coming up.
    """
    pname = current_process().name
    # Real-time scheduling, applied by the child itself: sched_setscheduler
    # acts on the calling process, and doing it here removes the race with
    # the first ticks that setting affinity from the parent after spawn left.
    if core is not None:
        pin_to_core(core, rt_priority if rt_priority is not None else 80)
    rt = params.RUNTIME           # runtime knobs

    completion_stage = 0

    def terminate_gracefully():
        gc.enable()  # undo the RT gc freeze unconditionally
        if completion_stage >= 1:
            for lg in (state_log, torque_log, timing_log):
                lg.save(run_dir)
            imu.close()
            # TerminalOperator puts the tty in cbreak; without this the shell is
            # left with no echo and no line buffering after every run.
            operator.close()
        if is_verbose:
            print(f"{pname}: terminated gracefully.")

    # A SIGTERM/SIGINT/exception sets stop_event and still runs teardown.
    with ExIntGuard(pname, stop_event, terminate_gracefully):
        # --- stage 1: init hardware AFTER spawn ------------------------------
        imu = IMUInterface()   # ports/baud default to params.IMU
        imu.connect()
        eskf = ESKF()
        hip_accel = BackwardDifference()
        pc = ProsthesisControl()
        # Operator intent. TerminalOperator needs a tty; in the spawned control
        # process stdin usually is not one, so it reports .available and we fall
        # back to the scripted source -- which is also what any headless or
        # replay run wants, because it is the only deterministic one.
        operator = TerminalOperator()
        if not operator.available:
            operator = TimedOperator(walk_at=rt.bench_walk_at,
                                     swing_at=rt.bench_swing_at,
                                     swing_duration=rt.bench_swing_duration)
            print(f"[{pname}] operator: timed  walk@{rt.bench_walk_at}s  "
                  f"swings@{list(rt.bench_swing_at)}s "
                  f"({rt.bench_swing_duration}s each)")
        state = ProsthesisState()   # this process's OWN local bus

        # tmax counts from the first ACTUATING tick, not from t0, so warmup
        # does not eat the run. Both processes detect that moment independently
        # -- control from its own FSM, low-level from the mode on the command it
        # receives -- which differ by one IPC hop (sub-ms) and need no extra
        # shared state. `max_arm_wait_s` bounds a run that never arms.
        num_cycles = floor((rt.max_arm_wait_s + tmax) / dt) + 1
        t_armed = None            # t of the first actuating tick
        state_log = Recorder("control_prosthesis_state")  # full ProsthesisState (incl. hip_*)
        torque_log = Recorder("control_torque")   # TorqueDesired each tick
        timing_log = Recorder("control_idle")     # LoopStats (idle margin + measured dt)

        completion_stage = 1
        # restore the terminal on the way out if we grabbed it

        # --- RT gc hygiene (mirrors main.py) ---------------------------------
        gc.collect()
        gc.freeze()
        gc.disable()

        # --- stamp the shared start origin (no barrier; sole stamper) --------
        # Low-level is spinning on `while start_time == 0`, heartbeating the
        # EtherCAT bus to hold the drives in OP. Stamping here -- after the slow
        # IMU init -- releases it, and both loops run from this common origin.
        with start_time.get_lock():
            start_time.value = perf_counter()
        t0 = start_time.value

        # --- soft-realtime loop ---------------------------------------------
        clock = t0
        t_prev = t0 - dt
        cycle = 0
        while cycle < num_cycles:
            now = perf_counter()
            t = now - t0
            dt_meas = now - t_prev
            t_prev = now

            imu_frame = imu.read()

            # adopt the full proprioceptive state off the buffer as this tick's
            # base, BEFORE the ESKF (its loop closures read joint velocities off
            # the bus). hip_* arrive as zero and are overlaid by eskf.update below;
            # every other physical field (motor, spring, torques, GRF) carries
            # through to the controller + the log. .seq is available for staleness.
            ms = buffer.latest()
            if ms is not None:
                state = ms.state

            # --- inline ESKF step (was ProsthesisStateEstimator._apply_eskf) ---
            # pass this loop's own clock t (shares the IMU timebase); dt_meas
            # drives the filter's integration + the hip_accel diff.
            eskf.update(imu_frame, state, t, dt_meas)
            if eskf.initialized:
                state.hip_accel = hip_accel.compute(state.hip_velocity, dt_meas)

            # --- mode FSM + high-level policy (co-located, no extra hop) -------
            # control OWNS the mode; the low level derives `actuate` and the
            # knee extension-stop suppression from it.
            op = operator.read(t)
            desired = pc.compute(state, t, op, dt_meas)   # TorqueDesired
            state.standing = pc.mode is Mode.STANDING     # for logging/consumers
            cmd_buf.publish(desired, pc.mode)             # -> B3, down to low-level

            # arm the run window on the first actuating tick, then end tmax
            # later. Checked after logging (below) so the last tick is recorded,
            # matching low-level.
            if t_armed is None and pc.mode.actuate:
                t_armed = t
                print(f"[{pname}] armed at t={t:.2f}s ({pc.mode.name}); "
                      f"running {tmax}s")
            armed_expired = t_armed is not None and t - t_armed >= tmax

            if stop_event.is_set():
                return

            clock += dt
            idle = clock - perf_counter()
            state_log.append(t, state)
            torque_log.append(t, desired)
            timing_log.append(t, LoopStats(idle, dt_meas))
            sleep(idle) if idle >= 0.0 else sleep(0.0)

            if armed_expired:
                break

            cycle += 1
    # (ExIntGuard runs terminate_gracefully on exit -- no finally needed)
