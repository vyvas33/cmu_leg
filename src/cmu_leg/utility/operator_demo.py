"""Drive the whole high-level stack from operator keypresses, on real data.

Replays the Oct-2025 walking trial through
    hall GRFs -> contact -> mode FSM -> gait FSM -> torque
and prints a live status line, so you can watch the mode transitions actually
refuse or permit as you press keys.

Keys (when run in a terminal):
    s   request standing
    w   request walking   -- watch it refuse until the foot is UNLOADED
    z   toggle hold
    q   panic: hold, latched
Ctrl-C exits.

Without a tty it falls back to a scripted TimedOperator and narrates the same
sequence, so it works headless and in CI.

Run:  uv run python -m cmu_leg.utility.operator_demo
      uv run python -m cmu_leg.utility.operator_demo --scripted   (force scripted)
      uv run python -m cmu_leg.utility.operator_demo --speed 4    (4x real time)
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from cmu_leg.control.high_level_control import ProsthesisControl
from cmu_leg.runtime.operator import TerminalOperator, TimedOperator
from cmu_leg.structures.commands import Mode
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.config.params import CONTACT, COMMON, MODE
from cmu_leg.control.utils import clamp

MAT = (Path(__file__).parent.parent / "tests" / "data"
       / "gaitData20251015_test7_NN_vicon.mat")
COL = dict(knee=7, knee_vel=8, ankle=15, ankle_vel=16, hip=74, toe=22, heel=23,
           time=107)
WALK_START_S = 16.5


def load():
    base = sio.loadmat(str(MAT), squeeze_me=True,
                       struct_as_record=False)["scope_data"].base.data
    t = base[:, COL["time"]]
    idx = np.flatnonzero(t >= WALK_START_S)[::2]      # -> 500 Hz
    return {k: base[idx, c] for k, c in COL.items()}


class ContactFromLog:
    """The estimator's contact logic, fed the log's already-filtered GRFs.

    Same threshold + live zero as ProprioceptiveEstimator._apply_contact; the
    estimator itself needs a RawEtherCATFrame, which we do not have here.
    """

    def __init__(self, toe0, heel0):
        self.toe_base, self.heel_base = toe0, heel0

    def update(self, toe, heel, dt) -> bool:
        contact = (abs(toe - self.toe_base) > CONTACT.toe_threshold
                   or abs(heel - self.heel_base) > CONTACT.heel_threshold)
        if not contact:
            a = clamp(dt / max(CONTACT.baseline_tau, dt), 0.0, 1.0)
            self.toe_base += a * (toe - self.toe_base)
            self.heel_base += a * (heel - self.heel_base)
        return contact


BAR = {Mode.INITIALIZING: "\033[90m", Mode.STANDING: "\033[36m",
       Mode.WALKING: "\033[32m", Mode.HOLD: "\033[31m"}
RESET = "\033[0m"


def main(argv):
    scripted = "--scripted" in argv
    speed = 1.0
    if "--speed" in argv:
        speed = float(argv[argv.index("--speed") + 1])

    sig = load()
    n = len(sig["time"])
    dt = 1.0 / COMMON.control_sample_rate

    operator = None if scripted else TerminalOperator()
    if operator is None or not operator.available:
        # Standing is the resting request, so nothing to schedule for it; the
        # script just asks to walk (while still LOADED -- refused until the foot
        # lifts) and later drops to hold.
        # walk_at must clear MODE.enable_dwell: standing has to happen first,
        # because INITIALIZING -> WALKING is not a transition (same as the
        # baseline -- you reach walking only from standing).
        operator = TimedOperator(walk_at=MODE.enable_dwell + 1.0, hold_at=12.0)
        print(f"[demo] scripted: standing once the {MODE.enable_dwell}s dwell "
              f"elapses  request-walk@{MODE.enable_dwell + 1.0}s  hold@12.0s")

    pc = ProsthesisControl()
    pc._standing._verified = pc._walking._verified = True
    pc._walking._stance_ctrl._verified = True

    contact_det = ContactFromLog(sig["toe"][0], sig["heel"][0])
    state = ProsthesisState()
    t0 = time.perf_counter()
    prev_mode = None
    events = []
    live = sys.stdout.isatty()   # the \r status line needs a terminal

    try:
        for i in range(n):
            t = i * dt
            state.knee_angle = sig["knee"][i]
            state.ankle_angle = sig["ankle"][i]
            state.knee_velocity = sig["knee_vel"][i]
            state.ankle_velocity = sig["ankle_vel"][i]
            state.hip_angle = sig["hip"][i]
            state.contact = contact_det.update(sig["toe"][i], sig["heel"][i], dt)
            # No EtherCAT here, so stand in for the drives: they would be
            # commanded at low_warmup_s and confirm OPERATION_ENABLED shortly
            # after. Without this the FSM correctly refuses to leave
            # INITIALIZING, which is the gate doing its job.
            state.control_enabled = True

            tq = pc.compute(state, t, operator.read(t), dt)

            if pc.mode is not prev_mode:
                events.append((t, prev_mode, pc.mode, state.contact))
                if not live:
                    src = prev_mode.name if prev_mode else "(start)"
                    print(f"  t={t:6.2f}s  {src:<13} -> {pc.mode.name:<13}"
                          f"  contact={'LOADED' if state.contact else 'unloaded'}")
                prev_mode = pc.mode

            if live and i % 10 == 0:
                phase = ("stance" if pc._walking._gait.in_stance else "swing "
                         ) if pc.mode is Mode.WALKING else "  --  "
                sys.stdout.write(
                    f"\r t={t:6.2f}s  {BAR[pc.mode]}{pc.mode.name:<13}{RESET}"
                    f" contact={'LOAD' if state.contact else '    '}"
                    f"  phase={phase}"
                    f"  knee={tq.knee_desired:8.2f}  ankle={tq.ankle_desired:8.2f} Nm ")
                sys.stdout.flush()

            target = t0 + t / speed
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
    except KeyboardInterrupt:
        pass
    finally:
        operator.close()
        if live:
            print("\n\nmode transitions:")
            for t, a, b, c in events:
                src = a.name if a else "(start)"
                print(f"  t={t:6.2f}s  {src:<13} -> {b.name:<13}"
                      f"  contact={'LOADED' if c else 'unloaded'}")


if __name__ == "__main__":
    main(sys.argv[1:])
