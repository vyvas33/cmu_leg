"""
Estimator process body. Runs as a child process.

Owns the two USB IMUs and the ESKF (and, in later phases, the whole high-level
controller). Each tick: reads the IMUs, pulls the latest full ProsthesisState off
the shared B1 StateBuffer, runs the ESKF to overlay hip_*, and logs the complete
state. It NEVER runs proprioceptive estimation -- the physical state comes from
the buffer, already assembled + differentiated by the low-level process.

Symmetric to runtime/low_level.py: a run() Process target, hardware initialized
AFTER spawn (USB handles can't be pickled), a completion_stage-guarded teardown,
the same start Barrier for time-sync, and an integer-cycle soft-realtime loop.
"""

import gc
from dataclasses import dataclass
from math import floor
from time import perf_counter, sleep
from multiprocessing import current_process

from cmu_leg.hardware.imu_io import IMUInterface
from cmu_leg.estimation.eskf.eskf import ESKF
from cmu_leg.control.utils import BackwardDifference
from cmu_leg.types.state import ProsthesisState
from cmu_leg.runtime.buffer import StateBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.utility.recorder import Recorder


@dataclass
class LoopStats:
    """Per-tick timing trace for the online idle-margin / loop-period plots.
    Recorder logs dataclasses, so the two scalars need this wrapper."""
    idle: float
    dt: float


def run(dt, tmax, buffer, state_buf, start_barrier, stop_event, start_time,
        run_dir, is_verbose=False):
    """Estimator child process entry point (the Process target).

    Args:
        dt (float): Nominal loop period (s) -- steps the sleep-to-clock ONLY.
            Bring-up at the low-level rate first, then drop to ~500 Hz.
        tmax (float): Max elapsed time (s); fixes the integer cycle count.
        buffer (StateBuffer): B1, low-level -> estimator. Sole reader; carries
            the full proprioceptive ProsthesisState (hip_* zero until we overlay).
        state_buf (StateBuffer): B2, estimator -> high-level. Sole writer.
        start_barrier (multiprocessing.Barrier): Time-sync with the low-level
            process; its action stamps the shared start_time.
        stop_event (multiprocessing.Event): Graceful-shutdown request.
        start_time (multiprocessing.Value): Shared 'd' synced perf_counter origin,
            read AFTER start_barrier.wait().
        run_dir (str | Path): Log directory, created ONCE by the parent so both
            children's CSVs land together and can be joined on `t`.
        is_verbose (bool): Verbose prints.

    No init_evnt here: nothing downstream is gated on the estimator coming up.
    """
    pname = current_process().name

    completion_stage = 0

    def terminate_gracefully():
        gc.enable()  # undo the RT gc freeze unconditionally
        if completion_stage >= 1:
            for lg in (state_log, timing_log):
                lg.save(run_dir)
            imu.close()
        if is_verbose:
            print(f"{pname}: terminated gracefully.")

    # A SIGTERM/SIGINT/exception sets stop_event and still runs teardown.
    with ExIntGuard(pname, stop_event, terminate_gracefully):
        # --- stage 1: init hardware AFTER spawn ------------------------------
        imu = IMUInterface()   # ports/baud default to params.IMU
        imu.connect()
        eskf = ESKF()
        hip_accel = BackwardDifference()
        state = ProsthesisState()   # this process's OWN local bus

        num_cycles = floor(tmax / dt) + 1
        state_log = Recorder("est_prosthesis_state")  # full ProsthesisState (incl. hip_*)
        timing_log = Recorder("est_idle")    # LoopStats (idle margin + measured dt)

        completion_stage = 1

        # --- RT gc hygiene (mirrors main.py) ---------------------------------
        gc.collect()
        gc.freeze()
        gc.disable()

        # --- sync start with the low-level process (two-phase, see low_level) -
        if start_barrier.wait() == 0:
            with start_time.get_lock():
                start_time.value = perf_counter()
        start_barrier.wait()
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
            # every other physical field (motor, spring, torques, GRF) now carries
            # through to B2 + the log. .seq is available for staleness checks.
            ms = buffer.latest()
            if ms is not None:
                state = ms.state

            # --- inline ESKF step (was ProsthesisStateEstimator._apply_eskf) ---
            # pass the estimator's OWN loop clock t (shares the IMU timebase);
            # dt_meas drives the filter's integration + the hip_accel diff.
            eskf.update(imu_frame, state, t, dt_meas)
            if eskf.initialized:
                state.hip_accel = hip_accel.compute(state.hip_velocity, dt_meas)

            # the full physical state is now assembled -> publish to high-level (B2)
            state_buf.publish(state)

            if stop_event.is_set():
                return

            clock += dt
            idle = clock - perf_counter()
            state_log.append(t, state)
            timing_log.append(t, LoopStats(idle, dt_meas))
            sleep(idle) if idle >= 0.0 else sleep(0.0)

            cycle += 1
    # (ExIntGuard runs terminate_gracefully on exit -- no finally needed)
