"""Replay ImpedanceStance over a real walking trial.

Drives `control.high_level.walking.impedance_stance.ImpedanceStance` with the
knee/ankle angles and velocities logged in a baseline level-walking trial, and
plots the torque it would have commanded.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
This trial ran the NEUROMUSCULAR controller, not impedance, so the logged
torques are not a reference for our output -- there is no recorded impedance
torque on this leg to diff against. What the replay checks is behaviour on real
kinematics:
  * does the 3-phase FSM advance 1->2->3 exactly once per stance, at sensible
    times within the step?
  * are the torque magnitudes physically plausible and continuous?
  * do the signs make sense (knee extension under load, ankle plantarflexing
    through push-off)?
The logged NM torque is plotted only as scale context, clearly labelled.

Data: data/gaitData20251015_test7_NN_vicon.mat (Oct 2025, CMU leg, 1 kHz, ~31 s).
Trial structure: 0-4 s pre-enable (knee frozen at 3.44 rad), 4-16 s standing,
~16.5-31.7 s walking.

Signal columns (0-based into scope_data.base.data), triangulated from the
embedded signalNames:

    knee angle [rad] ......... 7      knee velocity [rad/s] .... 8
    ankle angle [rad] ........ 18     ankle velocity [rad/s] ... 19
    stance flag (State Machine/p1) .. 29
    standing flag (State Machine/p2)  30
    knee torque (NM, logged) . 100    ankle torque (NM, logged)  105
    time [s] ................. 107

Cols 100/105 are the JOINT-side torque commands: they range to exactly the
+-100 Nm / +-150 Nm in KneeParams/AnkleParams.torque_max, i.e. they are the
saturated joint commands. Cols 99/104 are motor current [A], not torque.

`Ankle Angle` emits two conversions of the same encoder counts: a linear
counts-to-rad gain plus `-ankle_angle_offset` (the joint angle, cols 18/19/20)
and a polynomial fit with error correction (the spring angle, cols 15/16/17).
Controllers take the joint angle, so use 18/19/20; `encoders.py` draws the same
split into ankle_angle vs ankle_spring_angle.

`stance` is taken from the baseline's own state machine rather than
re-derived from the halls, so this exercises the controller alone and is not
contaminated by contact-detection error.

Run:  uv run python -m cmu_leg.tests.impedance_stance_replay
Writes impedance_stance_replay.png next to this file.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.control.high_level.walking.impedance_stance import ImpedanceStance
from cmu_leg.types.state import ProsthesisState
from cmu_leg.config.params import KNEE, ANKLE

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = HERE / "impedance_stance_replay.png"

COL = dict(knee_ang=7, knee_vel=8, ankle_ang=18, ankle_vel=19,
           stance=29, standing=30, knee_tq=100, ankle_tq=105, time=107)

WALK_START_S = 16.5     # standing ends / walking begins


def load_walking():
    """Return the walking-only slice of the trial as a dict of arrays."""
    base = sio.loadmat(str(MAT), squeeze_me=True,
                       struct_as_record=False)["scope_data"].base.data
    t = base[:, COL["time"]]
    w = t >= WALK_START_S
    return {k: base[w, c] for k, c in COL.items()}


def replay(sig, ctrl=None):
    """Run the controller over the trial. Returns torques and phase per sample.

    Mirrors what the walking FSM will do: reset on entry to stance, run while
    in stance, command zero in swing (the swing controller owns that half).
    """
    ctrl = ctrl or ImpedanceStance()
    ctrl._verified = True          # replay only; gate stays False in the repo

    n = len(sig["time"])
    knee_tq = np.zeros(n)
    ankle_tq = np.zeros(n)
    phase = np.full(n, np.nan)

    state = ProsthesisState()
    prev_stance = False
    for i in range(n):
        stance = sig["stance"][i] > 0.5
        state.knee_angle = sig["knee_ang"][i]
        state.ankle_angle = sig["ankle_ang"][i]
        state.knee_velocity = sig["knee_vel"][i]
        state.ankle_velocity = sig["ankle_vel"][i]

        if stance and not prev_stance:
            ctrl.reset(state, sig["time"][i])
        prev_stance = stance

        if stance:
            tq = ctrl.compute(state, sig["time"][i])
            knee_tq[i], ankle_tq[i] = tq.knee_desired, tq.ankle_desired
            phase[i] = ctrl._phase
    return knee_tq, ankle_tq, phase


def report(sig, knee_tq, ankle_tq, phase):
    """Print the per-step FSM summary and the torque stats."""
    t = sig["time"]
    st = sig["stance"] > 0.5
    starts = np.flatnonzero(np.diff(st.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(st.astype(int)) == -1) + 1
    if ends.size and starts.size and ends[0] < starts[0]:
        ends = ends[1:]
    n = min(len(starts), len(ends))

    print(f"\n{len(starts)} stance periods in {t[-1]-t[0]:.1f}s of walking\n")
    print(f"{'step':>4}{'dur[s]':>8}{'reached':>9}{'t@1->2':>9}{'t@2->3':>9}"
          f"{'|knee|max':>11}{'|ank|max':>10}")
    print("-" * 60)
    reached = []
    for k in range(n):
        s, e = starts[k], ends[k]
        ph = phase[s:e]
        ph = ph[~np.isnan(ph)]
        if ph.size == 0:
            continue
        top = int(ph.max())
        reached.append(top)
        i12 = np.flatnonzero(ph >= 1)
        i23 = np.flatnonzero(ph >= 2)
        f12 = (i12[0] / len(ph)) if i12.size else np.nan
        f23 = (i23[0] / len(ph)) if i23.size else np.nan
        print(f"{k+1:4d}{t[e]-t[s]:8.3f}{top+1:9d}"
              f"{f12:9.2f}{f23:9.2f}"
              f"{np.abs(knee_tq[s:e]).max():11.3f}{np.abs(ankle_tq[s:e]).max():10.3f}")
    print("\n(t@ columns are the fraction of stance elapsed at each transition)")
    if reached:
        print(f"top phase reached: min={min(reached)+1} max={max(reached)+1}")
    print(f"\nour torque   knee [{knee_tq.min():7.3f}, {knee_tq.max():7.3f}] Nm"
          f"   ankle [{ankle_tq.min():7.3f}, {ankle_tq.max():7.3f}] Nm")
    print(f"logged (NM)  knee [{sig['knee_tq'].min():7.3f}, {sig['knee_tq'].max():7.3f}] Nm"
          f"   ankle [{sig['ankle_tq'].min():7.3f}, {sig['ankle_tq'].max():7.3f}] Nm")


def plot(sig, knee_tq, ankle_tq, phase):
    t = sig["time"]
    st = sig["stance"] > 0.5
    fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    def shade(a):
        starts = np.flatnonzero(np.diff(st.astype(int)) == 1) + 1
        ends = np.flatnonzero(np.diff(st.astype(int)) == -1) + 1
        for s, e in zip(starts, ends[ends > starts[0]] if starts.size else []):
            a.axvspan(t[s], t[e], color="0.85", zorder=0)

    ax[0].plot(t, sig["knee_ang"], label="knee angle")
    ax[0].plot(t, sig["ankle_ang"], label="ankle angle")
    ax[0].axhline(0.18, ls=":", c="C0", lw=1, label="thr 1->2 (knee 0.18)")
    ax[0].axhline(0.05, ls=":", c="C1", lw=1, label="thr 2->3 (ankle 0.05)")
    ax[0].set_ylabel("angle [rad]"); shade(ax[0])
    ax[0].legend(ncol=4, fontsize=8); ax[0].set_title(
        "ImpedanceStance replayed on real walking kinematics (shaded = stance)")

    ax[1].step(t, phase + 1, where="post", c="C3")
    ax[1].set_ylabel("stance phase"); ax[1].set_yticks([1, 2, 3])
    ax[1].set_ylim(0.5, 3.5); shade(ax[1])

    ax[2].plot(t, knee_tq, c="C0", label="ours (impedance)")
    ax[2].plot(t, sig["knee_tq"], c="0.5", lw=0.8, label="logged NM joint cmd (different controller)")
    for s_ in (+1, -1):
        ax[2].axhline(s_*KNEE.torque_max, ls="--", c="r", lw=0.8)
    ax[2].set_ylabel("knee torque [Nm]"); shade(ax[2])
    ax[2].legend(fontsize=8, loc="lower left")

    ax[3].plot(t, ankle_tq, c="C1", label="ours (impedance)")
    ax[3].plot(t, sig["ankle_tq"], c="0.5", lw=0.8, label="logged NM joint cmd (different controller)")
    for s_ in (+1, -1):
        ax[3].axhline(s_*ANKLE.torque_max, ls="--", c="r", lw=0.8)
    ax[3].set_ylabel("ankle torque [Nm]"); ax[3].set_xlabel("time [s]")
    shade(ax[3]); ax[3].legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"\nwrote {PNG}")


if __name__ == "__main__":
    sig = load_walking()
    knee_tq, ankle_tq, phase = replay(sig)
    report(sig, knee_tq, ankle_tq, phase)
    plot(sig, knee_tq, ankle_tq, phase)
