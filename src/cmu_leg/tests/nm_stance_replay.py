"""Replay NeuromuscularStance over the baseline's own NM walking trial.

Unlike the impedance replay, this log was produced BY the neuromuscular
controller, so the logged joint torques are a real reference: same controller,
same data, any difference is our port.

Reference signals -- pick deliberately, they are NOT interchangeable:

  scope_data.nm[:,0], [:,1]   `NM stance Control/Gain`,`Gain1`. These take
      Muscle Models output directly into the Bus Creator, PARALLEL to
      torque_transformation -- a logging branch. So they are the HUMAN-frame
      torque x body weight, pre-transform. Use these to validate geometry +
      muscle + reflex without the transform in the way.
  base[:,96], [:,101]         `Product4/s1,s2`, the high-level desired torque
      after the stance/swing mix and the mixing block's body-weight gain, i.e.
      POST-transform. Use these to compare torque maps.
  base[:,100], [:,105]        low-level joint command, downstream of
      conditioning and saturation. Too far downstream to be a reference.

The KNEE reference contains kp_stop_nm / kd_stop_nm / kd_damper_nm, which are
knee-only and which we have not ported (the baseline sums them inside Muscle
Models, so they also pass through the Jacobian; ours would add them outside).
That is why the knee comparison is contaminated and the ankle is clean.

Also runs both candidate torque maps -- the baseline's M and the
virtual-work-consistent M^T -- since the whole point of contention is which one
the logged data actually came from.

`Ankle Angle` emits two conversions of the same encoder counts: a linear
counts-to-rad gain plus `-ankle_angle_offset` (the joint angle, cols 18/19/20)
and a polynomial fit with error correction (the spring angle, cols 15/16/17).
Controllers take the joint angle, so use 18/19/20; `encoders.py` draws the same
split into ankle_angle vs ankle_spring_angle.

Data + column map: see tests/impedance_stance_replay.py.

Run:  uv run python -m cmu_leg.tests.nm_stance_replay
Writes nm_stance_replay.png next to this file.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cmu_leg.control.high_level.walking.neuromuscular_stance as nms
from cmu_leg.control.high_level.walking.neuromuscular_stance import NeuromuscularStance
from cmu_leg.types.state import ProsthesisState

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = HERE / "nm_stance_replay.png"

COL = dict(knee_ang=7, knee_vel=8, ankle_ang=18, ankle_vel=19, hip_ang=74,
           stance=29, post_knee=96, post_ankle=101, time=107)
NMCOL = dict(hum_knee=0, hum_ankle=1,
             act_vas=6, act_gas=14, act_sol=23,
             lce_vas=3, lce_gas=11, lce_sol=20,
             lmtc_vas=7, lmtc_gas=15, lmtc_sol=24)

WALK_START_S = 16.5
DECIMATE = 2          # log is 1 kHz; the controller runs at 500 Hz


def load_walking():
    sd = sio.loadmat(str(MAT), squeeze_me=True,
                     struct_as_record=False)["scope_data"]
    base, nmd = sd.base.data, sd.nm.data
    t = base[:, COL["time"]]
    w = np.flatnonzero(t >= WALK_START_S)[::DECIMATE]
    sig = {k: base[w, c] for k, c in COL.items()}
    sig.update({k: nmd[w, c] for k, c in NMCOL.items()})
    return sig


def replay(sig, transpose):
    """Run the controller over the trial with the chosen torque map."""
    nms._TORQUE_MAP_TRANSPOSE = transpose
    ctrl = NeuromuscularStance()
    ctrl._verified = True

    n = len(sig["time"])
    knee_tq = np.zeros(n)
    ankle_tq = np.zeros(n)
    act = np.zeros((n, 3))
    hum = np.zeros((n, 2))

    state = ProsthesisState()
    prev_stance = False
    for i in range(n):
        stance = sig["stance"][i] > 0.5
        state.knee_angle = sig["knee_ang"][i]
        state.ankle_angle = sig["ankle_ang"][i]
        state.knee_velocity = sig["knee_vel"][i]     # the stop/damper needs it
        state.hip_angle = sig["hip_ang"][i]          # late-stance VAS suppression
        state.ankle_velocity = sig["ankle_vel"][i]
        if stance and not prev_stance:
            ctrl.reset(state, sig["time"][i])
        prev_stance = stance
        if stance:
            tq = ctrl.compute(state, sig["time"][i])
            knee_tq[i], ankle_tq[i] = tq.knee_desired, tq.ankle_desired
            phi = ctrl._to_human_angles(state.knee_angle, state.ankle_angle)
            tk = ctrl._stop_and_damper(state.knee_angle, state.knee_velocity)
            ta = 0.0
            for m in ctrl.muscles:
                rk, ra = ctrl._get_moment_arm(m, phi)
                F = m.get_muscle_force(m.l_ce, m.act, m.l_m)
                tk += rk * F; ta += ra * F
            hum[i] = [tk * ctrl._body_weight, ta * ctrl._body_weight]
        act[i] = [m.act for m in ctrl.muscles]
    return knee_tq, ankle_tq, act, hum


def stance_mask(sig):
    """Samples inside COMPLETE stance periods only.

    The raw stance flag stays true through the tail of the trial, after walking
    stops -- a long non-walking segment that drags every metric down (it cost
    ~0.08 on the VAS correlation). Only closed heel-strike -> toe-off intervals
    count.
    """
    st = sig["stance"] > 0.5
    starts = np.flatnonzero(np.diff(st.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(st.astype(int)) == -1) + 1
    ends = ends[ends > starts[0]]
    mask = np.zeros_like(st)
    for a, b in zip(starts, ends):
        mask[a:b] = True
    return mask


def report(sig, runs):
    st = stance_mask(sig)
    def sc(a, b): return np.corrcoef(a[st], b[st])[0, 1], np.sqrt(np.mean((a[st]-b[st])**2))

    k, an, act, hum = runs["M (baseline)"]
    print("A) HUMAN-frame torque vs nm[:,0]/[:,1] -- geometry + muscle + reflex + stop,")
    print("   independent of the torque map")
    for j, (lbl, ref) in enumerate((("knee", sig["hum_knee"]), ("ankle", sig["hum_ankle"]))):
        c, r = sc(hum[:, j], ref)
        print(f"     {lbl:<6} corr {c:6.3f}   rms {r:7.2f} Nm")

    print("\nB) POST-transform vs base[:,96]/[:,101] -- compares the two torque maps.")
    print("   The reference was GENERATED with M, so this measures fidelity to the")
    print("   baseline, not correctness. It cannot settle the transpose question.")
    print(f"     {'map':<14}{'joint':<7}{'corr':>8}{'rms[Nm]':>10}")
    for lbl, (k, an, _, _) in runs.items():
        for jn, ours, ref in (("knee", k, sig["post_knee"]),
                              ("ankle", an, sig["post_ankle"])):
            c, r = sc(ours, ref)
            print(f"     {lbl:<14}{jn:<7}{c:8.3f}{r:10.2f}")

    print("\nC) per-muscle internals (M run)")
    for j, n in enumerate(("VAS", "GAS", "SOL")):
        ca, ra = sc(act[:, j], sig[f"act_{n.lower()}"])
        print(f"     {n}: activation corr {ca:6.3f}  rms {ra:.4f}")


def plot(sig, runs):
    t, st = sig["time"], sig["stance"] > 0.5
    starts = np.flatnonzero(np.diff(st.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(st.astype(int)) == -1) + 1
    ends = ends[ends > starts[0]]
    k, an, act, hum = runs["M (baseline)"]

    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    def shade(a):
        for s_, e_ in zip(starts, ends):
            a.axvspan(t[s_], t[e_], color="0.88", zorder=0)

    ax[0].plot(t, sig["knee_ang"], label="knee"); ax[0].plot(t, sig["ankle_ang"], label="ankle")
    ax[0].axhline(0.2094, ls=":", c="r", lw=1, label="knee stop, 12 deg")
    ax[0].set_ylabel("angle [rad]"); shade(ax[0]); ax[0].legend(fontsize=8, ncol=3)
    ax[0].set_title("NeuromuscularStance vs the baseline's own NM trial (human frame)")

    ax[1].plot(t, sig["hum_knee"], c="0.4", lw=1.2, label="baseline nm[:,0]")
    ax[1].plot(t, hum[:, 0], c="C0", lw=1.0, label="ours")
    ax[1].set_ylabel("knee torque [Nm]"); shade(ax[1]); ax[1].legend(fontsize=8)

    ax[2].plot(t, sig["hum_ankle"], c="0.4", lw=1.2, label="baseline nm[:,1]")
    ax[2].plot(t, hum[:, 1], c="C1", lw=1.0, label="ours")
    ax[2].set_ylabel("ankle torque [Nm]"); shade(ax[2]); ax[2].legend(fontsize=8)

    for j, n in enumerate(("VAS", "GAS", "SOL")):
        ax[3].plot(t, sig[f"act_{n.lower()}"], c=f"C{j}", ls="--", lw=1.0, alpha=.7)
        ax[3].plot(t, act[:, j], c=f"C{j}", lw=1.0, label=f"{n} (dashed = baseline)")
    ax[3].set_ylabel("activation"); ax[3].set_xlabel("time [s]")
    ax[3].set_ylim(-0.05, 1.05); shade(ax[3]); ax[3].legend(fontsize=8, ncol=3)

    fig.tight_layout(); fig.savefig(PNG, dpi=110)
    print(f"\nwrote {PNG}")


if __name__ == "__main__":
    sig = load_walking()
    runs = {"M (baseline)": replay(sig, False), "M^T": replay(sig, True)}
    report(sig, runs)
    plot(sig, runs)
