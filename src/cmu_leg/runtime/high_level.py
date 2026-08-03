"""
High-level process body (P3). Runs as a child process.

The swappable control stage -- impedance today, a neuromuscular / NN controller
later. Isolated in its own process so a heavier controller cannot constrain the
estimator or the RT loop. Each tick: reads the full physical state off B2
(StateBuffer, from the estimator), decides the mode (standing/walking), runs the
active behavioral policy, and publishes the desired joint torques + mode flag to
B3 (CommandBuffer, down to the low-level process).

It holds NO hardware -- pure compute between two buffers. Same run() Process
target / guard / two-phase start-sync / integer-cycle scaffold as the others.
"""

import gc
from dataclasses import dataclass
from math import floor
from time import perf_counter, sleep
from multiprocessing import current_process

from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.types.state import ProsthesisState
from cmu_leg.runtime.buffer import StateBuffer, CommandBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.utility.recorder import Recorder

WARMUP_S = 5.0   # TODO: replace with the real mode FSM / supervisor


@dataclass
class LoopStats:
    """Per-tick timing trace for the online idle-margin / loop-period plots."""
    idle: float
    dt: float


def run(dt, tmax, state_buf, cmd_buf, start_barrier, stop_event, start_time,
        run_dir, is_verbose=False):
    """High-level child process entry point (the Process target).

    Args:
        dt (float): Nominal loop period (s) -- steps the sleep-to-clock.
        tmax (float): Max elapsed time (s); fixes the integer cycle count.
        state_buf (StateBuffer): B2, estimator -> high-level. Read only.
        cmd_buf (CommandBuffer): B3, high-level -> low-level. Write only.
        start_barrier (multiprocessing.Barrier): time-sync (now Barrier(3)).
        stop_event (multiprocessing.Event): graceful-shutdown request.
        start_time (multiprocessing.Value): shared synced perf_counter origin.
        run_dir (str | Path): shared log directory.
        is_verbose (bool): verbose prints.
    """
    pname = current_process().name

    completion_stage = 0

    def terminate_gracefully():
        gc.enable()
        if completion_stage >= 1:
            for lg in (torque_log, timing_log):
                lg.save(run_dir)
        if is_verbose:
            print(f"{pname}: terminated gracefully.")

    with ExIntGuard(pname, stop_event, terminate_gracefully):
        # --- init (no hardware; pure compute) --------------------------------
        pc = ProsthesisControl()
        state = ProsthesisState()   # working copy, overwritten from B2 each tick

        num_cycles = floor(tmax / dt) + 1
        torque_log = Recorder("high_torque")   # TorqueDesired each tick
        timing_log = Recorder("high_idle")     # LoopStats

        completion_stage = 1

        gc.collect()
        gc.freeze()
        gc.disable()

        # --- two-phase start sync (see low_level.py) -------------------------
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

            # adopt the estimator's full physical state (or hold last if none yet)
            ss = state_buf.latest()
            if ss is not None:
                state = ss.state

            # mode decision -- high-level OWNS `standing` (the estimator's copy is
            # inert). TODO: real mode FSM / supervisor; time-based stand-in now.
            state.standing = t > WARMUP_S
            state.standing = False # comment out line to send real torques.
            desired = pc.compute(state, t)              # TorqueDesired
            cmd_buf.publish(desired, state.standing)    # -> B3, down to low-level

            if stop_event.is_set():
                return

            clock += dt
            idle = clock - perf_counter()
            torque_log.append(t, desired)
            timing_log.append(t, LoopStats(idle, dt_meas))
            sleep(idle) if idle >= 0.0 else sleep(0.0)

            cycle += 1
    # (ExIntGuard runs terminate_gracefully on exit)
