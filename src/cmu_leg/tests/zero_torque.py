"""
Zero-command drive bring-up test (the "0 torque" test).

Purpose: validate the EtherCAT *write* path and the CiA-402 enable handshake on
real hardware, with the safest possible command -- all zeros -- so the drives
walk up to OPERATION_ENABLED while nothing is commanded to move. This is the
write-side counterpart to the read-only utility/ loggers.

  uv run python -m cmu_leg.tests.zero_torque

!!! SAFETY -- read before running !!!
  * Leg ON A STAND, e-stop in hand. This script ENERGIZES the motors.
  * Runtime mode is CSV (Cyclic Sync Velocity). A zero command means
    Target Velocity = 0, NOT zero motor effort: once enabled the joint
    actively HOLDS zero velocity (it goes STIFF and resists backdrive). It is
    not limp. "Nothing moves" is the expected, safe outcome -- not "free".
  * `enable` defaults False and the first write sends a shutdown control word,
    so the drives stay disabled until this script explicitly flips `enable`.

What it does, in three phases:
  1. WARM-UP  (enable=False): exchange zero frames to populate the interface's
     last-frame cache and confirm the bus/medulla are alive and the drives sit
     in a known, non-fault state -- BEFORE energizing.
  2. ENABLE   (enable=True): keep writing zeros; watch each drive climb the
     CiA-402 ladder to OPERATION_ENABLED; hold there confirming control_enabled.
  3. SHUTDOWN (enable=False): command shutdown for a few ticks, then close.

vkalva, June 2026
"""

import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.types.commands import DriveCommand
from cmu_leg.types.sensors import RawEtherCATFrame
from cmu_leg.hardware.ethercat_io import EtherCATInterface
from cmu_leg.hardware.elmos.state_machine import drive_state, DriveState
from cmu_leg.utility.utils import make_run_dir

PERIOD = 0.001          # target loop period (s) -> 1 kHz, matches DC SYNC0
WARMUP_SECS = 1.0       # disabled exchanges before enabling (cache + sanity)
ENABLE_SECS = 5.0       # time to climb to OPERATION_ENABLED and hold
SHUTDOWN_SECS = 0.5     # disabled exchanges to command shutdown before close

# Order the CiA-402 states so a plot shows the climb as a rising line.
_STATE_RANK = {
    DriveState.UNKNOWN: -1,
    DriveState.FAULT: 0,
    DriveState.FAULT_REACTION_ACTIVE: 1,
    DriveState.NOT_READY_TO_SWITCH_ON: 2,
    DriveState.SWITCH_ON_DISABLED: 3,
    DriveState.READY_TO_SWITCH_ON: 4,
    DriveState.SWITCHED_ON: 5,
    DriveState.QUICK_STOP_ACTIVE: 6,
    DriveState.OPERATION_ENABLED: 7,
}


def _states(frame: RawEtherCATFrame) -> tuple[DriveState, DriveState]:
    """Decode (knee, ankle) CiA-402 state from a frame's status words."""
    return drive_state(frame.knee_status_word), drive_state(frame.ankle_status_word)


def main():
    ec = EtherCATInterface("eth0")
    ec.connect()

    # Per-tick log. Add/trim columns as you decide what you want to inspect.
    log = {
        "t": [], "knee_state": [], "ankle_state": [],
        "drives_enabled": [], "medulla_alive": [], "control_enabled": [],
        "knee_motor_counts": [], "ankle_motor_counts": [],  # confirm nothing moves
        "idle_ms": [],
    }
    cmd = DriveCommand()  # all zeros -- never reassigned; this is the whole point

    def record(t: float, frame: RawEtherCATFrame, idle_ms: float) -> None:
        ks, as_ = _states(frame)
        log["t"].append(t)
        log["knee_state"].append(ks.name)
        log["ankle_state"].append(as_.name)
        log["drives_enabled"].append(ec.drives_enabled())
        log["medulla_alive"].append(ec.medulla_alive())
        log["control_enabled"].append(ec.control_enabled())
        log["knee_motor_counts"].append(frame.knee_motor_counts)
        log["ankle_motor_counts"].append(frame.ankle_motor_counts)
        log["idle_ms"].append(idle_ms)

    t0 = time.perf_counter()
    clk = t0
    reached_enabled = False

    def run_phase(duration: float) -> None:
        """Tick at 1 kHz for `duration`, writing the zero command each tick."""
        nonlocal clk, reached_enabled
        phase_start = time.perf_counter() - t0
        while (t := time.perf_counter() - t0) - phase_start < duration:
            clk += PERIOD
            frame = ec.write(cmd)                       # zeros + derived control word
            idle = clk - time.perf_counter()
            record(t, frame, idle * 1000.0)

            # CONFIRM: first transition into OPERATION_ENABLED on both drives.
            if not reached_enabled and ec.control_enabled():
                reached_enabled = True
                print(f"[{t:6.3f}s] control_enabled -> True (both drives OPERATION_ENABLED)")

            if idle > 0:
                time.sleep(idle)

    try:
        # --- Phase 1: WARM-UP, drives still disabled -------------------------
        # enable is already False. These exchanges populate ec._last_frame and
        # let us sanity-check the bus before energizing.
        print("warm-up (disabled)...")
        run_phase(WARMUP_SECS)

        # TODO/CONFIRM before enabling: decide your go/no-go gate here, e.g.
        #   - ec.medulla_alive() is True
        #   - neither drive is in FAULT / UNKNOWN
        # and abort (return) if not satisfied. Fill this in:
        ks, as_ = _states(ec._last_frame)
        print(f"pre-enable state: knee={ks.name} ankle={as_.name} "
              f"medulla_alive={ec.medulla_alive()}")
        # if <not safe>: print("aborting: ..."); return

        # --- Phase 2: ENABLE -------------------------------------------------
        print("enabling (watch the CiA-402 climb; nothing should move)...")
        ec.enable = True
        run_phase(ENABLE_SECS)

        if not reached_enabled:
            print("WARNING: never reached OPERATION_ENABLED -- inspect the state log")

    except KeyboardInterrupt:
        print("terminated early")
    finally:
        # --- Phase 3: SHUTDOWN ----------------------------------------------
        # Flip enable off and keep ticking so write() sends the shutdown control
        # word before we drop the bus. Do this even on KeyboardInterrupt.
        ec.enable = False
        try:
            run_phase(SHUTDOWN_SECS)
        finally:
            ec.close()
            out = make_run_dir("zero_torque")
            _dump_csv(log, out)
            _plot_states(log, out)
            _plot_idle(log["t"], log["idle_ms"], out)


def _dump_csv(log: dict, out) -> None:
    cols = list(log)
    path = out / "zero_torque.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(zip(*(log[c] for c in cols)))
    print("saved", path)


def _plot_states(log: dict, out) -> None:
    """Plot the CiA-402 state rank for both drives over time -- the climb to
    OPERATION_ENABLED should show as a step up to the top rank and stay there."""
    t = log["t"]
    knee = [_STATE_RANK[DriveState[s]] for s in log["knee_state"]]
    ankle = [_STATE_RANK[DriveState[s]] for s in log["ankle_state"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.step(t, knee, where="post", label="knee", linewidth=1.2)
    ax.step(t, ankle, where="post", label="ankle", linewidth=1.2)
    ax.set_yticks(sorted(_STATE_RANK.values()))
    ax.set_yticklabels([s.name for s in sorted(_STATE_RANK, key=_STATE_RANK.get)])
    ax.set_title("CiA-402 state climb", fontweight="bold")
    ax.set_xlabel("time [s]")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / "states.png"
    fig.savefig(png, dpi=110); plt.close(fig)
    print("saved", png)


def _plot_idle(t, idle_ms, out) -> None:
    """Loop slack before each 1 kHz tick, in ms. Dips below zero = overran the
    cycle budget (and the DC SYNC0 deadline)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, idle_ms, linewidth=0.7)
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1.0, label="overrun threshold")
    ax.set_title("Loop idle time", fontweight="bold")
    ax.set_xlabel("time [s]"); ax.set_ylabel("idle [ms]")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / "loop_idle.png"
    fig.savefig(png, dpi=110); plt.close(fig)
    print("saved", png)


if __name__ == "__main__":
    main()
