"""Replay Phase2 against the baseline's logged phase 2 command.

The hip signals the law sees are filtered inside `NN swing torque Control`
(2nd-order Butterworth, 50 Hz on the angle and 25 Hz on the velocity, at the
model's 1 kHz), so the raw `base` columns are reproduced through the same
filters here rather than fed in directly.

The entry latch is pinned from the log: slope_ini comes from `nntorque` col 19
and the starting knee angle from `knee_desired` on the tick before entry. That
latch belongs to the enclosing controller, which is not written yet, so pinning
it keeps this a test of the phase 2 law alone.

References, all `nntorque`:
    col 11  phase 2 knee command        col 14  slope
    col 12  phase 2 knee velocity       col 17  hip_vel after the floor
    col 19  slope_ini

Run:  uv run python -m cmu_leg.tests.nn_phase2_replay
Writes plots/nn_phase2_replay.png.

Authors: vkalva, Claude
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.tests import plot_path
from cmu_leg.config.params import COMMON, NNSWING
from cmu_leg.control.utils import Lowpass
from cmu_leg.control.high_level.walking.nn_swing import Phase2

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = plot_path("nn_phase2_replay.png")

BASE = dict(hip_angle=0, hip_vel=1)
NT = dict(phase2=3, knee=11, knee_vel=12, count=13, slope=14, knee_frozen=15,
          hip_frozen=16, hip_vel_frozen=17, slope_ini=19, knee_desired=23,
          time=25)

LOG_RATE = 1000.0
WALK_START_S = 16.5
HIP_ANGLE_CUTOFF = 50.0
HIP_VEL_CUTOFF = 25.0

# The model on disk postdates this trial by a month and has gained a 26 deg
# lower clamp on the hip angle feeding phase 2. Solving the fade equation for
# the hip angle the log must have used returns the unclamped signal, so the
# trial predates that block. Pinned off here, the same way the min-jerk replay
# pins its baseline gains.
CFG = replace(NNSWING, hip_angle_min=-10.0)


def load():
    sd = sio.loadmat(str(MAT), squeeze_me=True, struct_as_record=False)["scope_data"]
    base, nt = sd.base.data, sd.nntorque.data
    w = nt[:, NT["time"]] >= WALK_START_S
    sig = {k: nt[w, c].astype(float) for k, c in NT.items()}

    # the filters run from model start, so prime them on the whole record and
    # only then take the walking window
    for name, cut in (("hip_angle", HIP_ANGLE_CUTOFF), ("hip_vel", HIP_VEL_CUTOFF)):
        lp = Lowpass(cutoff_freq=cut, sample_rate=LOG_RATE, order=2)
        raw = base[:, BASE[name]].astype(float)
        lp.reset(raw[0])
        sig[name] = np.array([lp.compute(x) for x in raw])[w]
    return sig


def episodes(flag):
    d = np.diff(np.r_[0, flag != 0, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def replay_episode(sig, s, e):
    """Run Phase2 over one episode with the entry latch pinned from the log."""
    p = Phase2(cfg=CFG, common=replace(COMMON, control_sample_rate=LOG_RATE))
    t = sig["time"]
    hip_vel_entry = sig["hip_vel"][s]
    p.reset(sig["knee_desired"][s - 1],
            sig["slope_ini"][s] * hip_vel_entry, hip_vel_entry, t[s])

    cmd = np.empty(e - s)
    for i, k in enumerate(range(s, e)):
        cmd[i] = p.compute(t[k], sig["hip_angle"][k], sig["hip_vel"][k])
    return cmd


def main():
    sig = load()
    eps = episodes(sig["phase2"])
    print(f"{len(eps)} phase 2 episodes\n")
    print("   start     ticks   slope_ini   knee travel      max|err|     rms")

    out = []
    for s, e in eps:
        cmd = replay_episode(sig, s, e)
        ref = sig["knee"][s:e]
        err = np.degrees(np.abs(cmd - ref))
        print(f"   {sig['time'][s]:7.3f}  {e - s:5d}   {sig['slope_ini'][s]:+8.3f}"
              f"   {np.degrees(ref[0] - ref[-1]):6.1f} deg"
              f"   {err.max():9.4f} deg  {err.std():7.4f}")
        out.append((s, e, cmd, err))

    allerr = np.concatenate([o[3] for o in out])
    print(f"\n   all {len(allerr)} ticks: max {allerr.max():.4f} deg, "
          f"rms {allerr.std():.4f} deg")
    plot(sig, out)


def plot(sig, out):
    t = sig["time"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for i, (s, e, cmd, err) in enumerate(out):
        a = dict(label="baseline") if i == 0 else {}
        b = dict(label="ours") if i == 0 else {}
        ax[0].plot(t[s:e], np.degrees(sig["knee"][s:e]), "k", lw=1.3, **a)
        ax[0].plot(t[s:e], np.degrees(cmd), "C3--", lw=1.0, **b)
        ax[1].plot(t[s:e], err, "C3", lw=0.9)
    ax[0].set_ylabel("phase 2 knee command [deg]")
    ax[0].legend(loc="upper right")
    ax[1].set_ylabel("|error| [deg]")
    ax[1].set_xlabel("trial time [s]")
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"\nwrote {PNG}")


if __name__ == "__main__":
    main()
