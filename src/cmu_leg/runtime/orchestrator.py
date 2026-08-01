"""
Multiprocess orchestrator (the parent). Builds the shared IPC objects, spawns
the three child processes, and manages their lifecycle.

Processes (all async, each its own ~1 kHz loop):
    low-level  -- EtherCAT + ProprioceptiveEstimator; writes B1, reads B3
    estimator  -- ESKF; reads B1, assembles full state, writes B2
    high-level -- swappable controller; reads B2, writes B3

Buffers:
    B1 MedullaStateBuffer  low-level -> estimator
    B2 StateBuffer         estimator -> high-level  (full ProsthesisState)
    B3 CommandBuffer       high-level -> low-level  (torque + standing)

Startup handshake:
  1. spawn low-level; it inits EtherCAT and sets init_evnt.
  2. block on init_evnt (hardware confirmed up) -- only THEN spawn estimator +
     high-level.
  3. all three hit Barrier(3); the first stamps the shared start_time and every
     loop begins from that common origin (two-phase, see low_level.py).

The parent does not run a realtime loop and does not log; each child logs its own
trace into the shared run_dir. The parent monitors liveness and, on Ctrl-C /
SIGTERM / a child fault (stop_event), terminates everything gracefully.

Run:  uv run python -m cmu_leg.runtime.orchestrator
"""

import os
from multiprocessing import (Process, Event, Value, Barrier,
                             set_start_method, freeze_support)
from time import perf_counter

from cmu_leg.runtime.buffer import MedullaStateBuffer, StateBuffer, CommandBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.runtime import low_level, estimator, high_level
from cmu_leg.utility.utils import make_run_dir

DT_LOW = 1.0 / 1000    # low-level loop period (s)
DT_EST = 1.0 / 1000    # estimator loop period (s); drop to 1/500 after bring-up
DT_HIGH = 1.0 / 1000   # high-level loop period (s)
TMAX = 10.0            # experiment duration (s)
HW_INIT_TIMEOUT = 5.0  # max wait for the low-level hardware to come up (s)
CORES = (2, 3, 4)      # CPU affinity for (low-level, estimator, high-level); Linux only


def main():
    run_dir = make_run_dir("multiprocess")

    # --- shared IPC objects (built in the parent, passed to the children) ----
    b1 = MedullaStateBuffer()     # low-level  -> estimator
    b2 = StateBuffer()            # estimator  -> high-level
    b3 = CommandBuffer()          # high-level -> low-level
    stop_event = Event()          # any process -> shutdown everywhere
    init_evnt = Event()           # low-level -> "EtherCAT is up"
    start_time = Value("d", 0.0)  # synced perf_counter origin (barrier-stamped)
    start_barrier = Barrier(3)    # the three children; parent is not in it

    procs = []

    def terminate():
        """Set the stop flag and join the children; force-kill any straggler."""
        stop_event.set()
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join()
        print("\nMAIN: multiprocess control terminated gracefully.")

    with ExIntGuard("MAIN", stop_event, terminate, is_main_process=True):
        # --- spawn low-level, then wait for its hardware to come up ----------
        low_p = Process(
            name="LOWLEVEL", target=low_level.run,
            kwargs=dict(dt=DT_LOW, tmax=TMAX, buffer=b1, cmd_buf=b3,
                        start_barrier=start_barrier, stop_event=stop_event,
                        start_time=start_time, init_evnt=init_evnt,
                        run_dir=run_dir))
        low_p.start()
        procs.append(low_p)

        if not init_evnt.wait(timeout=HW_INIT_TIMEOUT):
            print("MAIN: low-level hardware init failed/timed out. Shutting down.")
            terminate()
            return

        # --- hardware confirmed up: spawn estimator + high-level -------------
        est_p = Process(
            name="ESTIMATOR", target=estimator.run,
            kwargs=dict(dt=DT_EST, tmax=TMAX, buffer=b1, state_buf=b2,
                        start_barrier=start_barrier, stop_event=stop_event,
                        start_time=start_time, run_dir=run_dir))
        est_p.start()
        procs.append(est_p)

        high_p = Process(
            name="HIGHLEVEL", target=high_level.run,
            kwargs=dict(dt=DT_HIGH, tmax=TMAX, state_buf=b2, cmd_buf=b3,
                        start_barrier=start_barrier, stop_event=stop_event,
                        start_time=start_time, run_dir=run_dir))
        high_p.start()
        procs.append(high_p)

        # --- pin to cores (Linux; no-op on macOS) ----------------------------
        if hasattr(os, "sched_setaffinity"):
            for p, core in zip(procs, CORES):
                try:
                    os.sched_setaffinity(p.pid, {core})
                except OSError as e:
                    print(f"MAIN: could not set affinity for {p.name}: {e}")

        # --- monitor until the children finish, fault, or we're interrupted --
        deadline = perf_counter() + TMAX + 1.0
        while any(p.is_alive() for p in procs) and perf_counter() < deadline:
            if stop_event.wait(timeout=0.5):
                break

    # ExIntGuard runs terminate() on the way out (normal exit or interrupt).


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()
