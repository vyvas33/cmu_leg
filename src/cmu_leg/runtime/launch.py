"""
Multiprocess launcher (the parent). Builds the shared IPC objects, spawns the
three child processes, and manages their lifecycle.

This is the multiprocess run entry point. It will eventually be folded into
main.py once main becomes the multiprocess entry (today main.py is the
single-process path), at which point this file goes away.

Processes (both async, each its own loop):
    low-level  -- EtherCAT + ProprioceptiveEstimator; writes B1, reads B3 (~1 kHz)
    control    -- ESKF + high-level controller (MERGED); reads B1, writes B3 (~500 Hz)

Buffers:
    B1 StateBuffer         low-level -> control  (full proprioceptive state)
    B3 CommandBuffer       control -> low-level  (torque + standing)

The estimator and high-level stages used to be two separate processes with a B2
StateBuffer between them; they were merged into `control` to cut the ESKF->
controller IPC hop that was destabilizing standing (see runtime/control.py).

Startup handshake:
  1. spawn low-level; it inits EtherCAT and sets init_evnt.
  2. block on init_evnt (hardware confirmed up) -- only THEN spawn control.
  3. control inits its IMUs (~seconds) and then stamps the shared start_time.
     Low-level is NOT blocked -- it heartbeats the EtherCAT bus (holding the drives
     in OP) and reads start_time once stamped, so both loops begin from the common
     origin without letting the bus go silent. No Barrier: control is the sole
     stamper now that it's the only other process.

The parent does not run a realtime loop and does not log; each child logs its own
trace into the shared run_dir. The parent monitors liveness and, on Ctrl-C /
SIGTERM / a child fault (stop_event), terminates everything gracefully.

Run:  uv run python -m cmu_leg.runtime.launch
"""

import os
import json
import subprocess
from dataclasses import asdict
from multiprocessing import (Process, Event, Value,
                             set_start_method, freeze_support)
from time import perf_counter

from cmu_leg.runtime.buffer import StateBuffer, CommandBuffer
from cmu_leg.runtime.guard import ExIntGuard
from cmu_leg.runtime import low_level, control
from cmu_leg.utility.utils import make_run_dir
import cmu_leg.config.params as params

# Loop periods derive from the config rates -- single source of truth. The
# control rate in particular is shared with the fixed-coefficient filters that
# tick in the control process (standing velocity lowpasses), so it must not drift
# from what those filters were designed at. Everything else (tmax, timeouts,
# cores, warmups, safety gates) is in params.RUNTIME.
DT_LOW = 1.0 / params.COMMON.sample_rate              # low-level loop period (s)
DT_CONTROL = 1.0 / params.COMMON.control_sample_rate  # control loop period (s)


def _check_warmups(rt):
    """Fail fast if the standing min-jerk would start before torque can flow.

    control_warmup_s (min-jerk start) must clear low_warmup_s (drive enable) plus
    the enable-walk + startup-mute lag, or the ramp elapses into a dead drive and
    the leg snaps to setpoint. This is the exact bug that produced the 11 Hz
    'oscillation'; the assertion makes it un-shippable rather than silent.
    """
    live = rt.low_warmup_s + rt.actuation_warmup_margin_s
    if rt.control_warmup_s < live:
        raise SystemExit(
            f"MAIN: RuntimeParams warmup desync -- control_warmup_s "
            f"({rt.control_warmup_s}s) must be >= low_warmup_s + "
            f"actuation_warmup_margin_s ({rt.low_warmup_s} + "
            f"{rt.actuation_warmup_margin_s} = {live}s). The standing min-jerk "
            f"would start before torque is deliverable and the leg would snap "
            f"instead of ramping. Raise control_warmup_s in params.RUNTIME.")


def _git_info():
    """Best-effort {commit, dirty} so a run's log pins the exact code. None on any
    failure (not a repo / git missing) -- never blocks a run."""
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _dump_config(run_dir):
    """Write config.json (git + the params this run used) into run_dir and echo a
    one-line summary, so every run is self-documenting -- no hand notes needed."""
    cfg = {
        "git": _git_info(),
        "runtime": asdict(params.RUNTIME),
        "common": asdict(params.COMMON),
        "standing": asdict(params.STANDING),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    rt, g = params.RUNTIME, cfg["git"]
    tag = (str(g["commit"])[:10] + (" (dirty)" if g["dirty"] else "")) if g["commit"] else "no-git"


def main():
    rt = params.RUNTIME
    _check_warmups(rt)
    run_dir = make_run_dir("multiprocess")
    _dump_config(run_dir)

    # --- shared IPC objects (built in the parent, passed to the children) ----
    b1 = StateBuffer()            # low-level -> control (full physical state)
    b3 = CommandBuffer()          # control   -> low-level (torque + standing)
    stop_event = Event()          # any process -> shutdown everywhere
    init_evnt = Event()           # low-level -> "EtherCAT is up"
    start_time = Value("d", 0.0)  # synced perf_counter origin (control stamps it)
    # No Barrier: control is the sole stamper. Low-level does NOT block on it -- it
    # must keep the EtherCAT bus alive (heartbeat) or the Elmo SM watchdog drops the
    # drives out of OP during control's IMU init. It reads the same start_time once
    # control stamps it after IMU bring-up.

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
            kwargs=dict(dt=DT_LOW, tmax=rt.tmax, buffer=b1, cmd_buf=b3,
                        stop_event=stop_event, start_time=start_time,
                        init_evnt=init_evnt, run_dir=run_dir))
        low_p.start()
        procs.append(low_p)

        if not init_evnt.wait(timeout=rt.hw_init_timeout):
            print("MAIN: low-level hardware init failed/timed out. Shutting down.")
            terminate()
            return

        # --- hardware confirmed up: spawn control (ESKF + high-level) --------
        ctrl_p = Process(
            name="CONTROL", target=control.run,
            kwargs=dict(dt=DT_CONTROL, tmax=rt.tmax, buffer=b1, cmd_buf=b3,
                        stop_event=stop_event, start_time=start_time,
                        run_dir=run_dir))
        ctrl_p.start()
        procs.append(ctrl_p)

        # --- pin to cores (Linux; no-op on macOS) ----------------------------
        if hasattr(os, "sched_setaffinity"):
            for p, core in zip(procs, rt.cores):
                try:
                    os.sched_setaffinity(p.pid, {core})
                except OSError as e:
                    print(f"MAIN: could not set affinity for {p.name}: {e}")

        # --- anchor the deadline to the control-stamped origin ----------------
        # Control stamps `start_time` only after its IMU init (~seconds), which
        # trails spawn. Base the liveness deadline on THAT origin, not the parent's
        # spawn time -- otherwise it fires ~(IMU-init) seconds early and cuts the
        # run short (each child runs exactly tmax/dt cycles from start_time).
        origin_timeout = perf_counter() + rt.hw_init_timeout + 10.0
        while start_time.value == 0.0 and perf_counter() < origin_timeout:
            if stop_event.wait(timeout=0.05):
                break
        t0 = start_time.value or perf_counter()

        # --- monitor until the children finish, fault, or we're interrupted --
        deadline = t0 + rt.tmax + 1.0
        while any(p.is_alive() for p in procs) and perf_counter() < deadline:
            if stop_event.wait(timeout=0.5):
                break

    # ExIntGuard runs terminate() on the way out (normal exit or interrupt).


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()
