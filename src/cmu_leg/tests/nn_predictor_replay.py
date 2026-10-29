"""Replay NeuralPredictor against the baseline's own logged network output.

The trial ran the NPC controller with the NN swing path live, and the `nn`
scope group logs both sides of the geometry transform, so two things can be
checked independently:

1. the network itself, against the logged MLP output (human frame)
2. the transform back to our leg, against the logged `nn_traj`

The spline is not checked here. Its matrix was verified directly against
MATLAB's spline() when tools/build_nn_weights.py built it (1.9e-14 over random
vectors), and the log does not carry the resampled trajectory, so comparing it
here would only be comparing our matmul to itself.

Inputs come straight off the log rather than through our ESKF, so this isolates
the predictor: a disagreement here is the network, the normalization, the
snapshot, or the geometry, never the estimator.

Two things about the baseline's inputs that our runtime does not reproduce:

  hip z carries a load-dependent correction (`Get Prosthesis State/Add`,
  base 86) that needs the rig's analog force plate. The NN does not use it --
  `data transfer` passes hip_pos through untouched -- and the numbers agree:
  raw gives 0.023 deg median against 0.068 with the correction. Settled; we
  could not read that force plate anyway.

  length_thigh is subject specific. params_new.m lists 0.55 to 0.71 for
  different people and our LEG.hip_to_knee is 0.66; the trial's subject is not
  recorded, so THIGH_CANDIDATES is swept and the best reported.

Column maps (0-based).
  base   0   ESKF/Gain2, = -hip_pitch, our hip_angle convention
         58  hip_pos x        60  hip_pos z (raw)
         61  knee_pos x       63  knee_pos z
         86  hip z after the load correction
         107 time
  nn     0   computation_signal, the network's own enable
         2..42   nn_traj, 41 points, our leg's frame
         43..83  raw MLP output, 41 points, human frame
  nntorque 23  knee_desired data store, the commanded knee

Run:  uv run python -m cmu_leg.tests.nn_predictor_replay
Writes plots/nn_predictor_replay.png.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.tests import plot_path
from cmu_leg.config.params import COMMON
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.control.high_level.walking import retargeting
from cmu_leg.control.high_level.walking import nn_predictor as npx

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = plot_path("nn_predictor_replay.png")

BASE = dict(hip_angle=0, hip_x=58, hip_z_raw=60, knee_x=61, knee_z=63,
            hip_z_comp=86, time=107)
NN_COMPUTE = 0
NN_LEG = slice(2, 43)          # 41 points, leg frame
NN_HUMAN = slice(43, 84)       # 41 points, human frame
NT_KNEE_DESIRED = 23

LOG_RATE = 1000.0
WALK_START_S = 16.5
THIGH_CANDIDATES = (0.55, 0.63, 0.66, 0.68, 0.71)
THIGH_LENGTH = 0.71   # params.m, "0.71 for Russell"


def load():
    sd = sio.loadmat(str(MAT), squeeze_me=True, struct_as_record=False)["scope_data"]
    base, nn, nt = sd.base.data, sd.nn.data, sd.nntorque.data
    t = base[:, BASE["time"]]
    w = t >= WALK_START_S
    sig = {k: base[w, c].astype(float) for k, c in BASE.items()}
    sig["knee_desired"] = nt[w, NT_KNEE_DESIRED].astype(float)
    sig["compute"] = nn[w, NN_COMPUTE].astype(float)
    sig["traj_leg"] = nn[w, NN_LEG].astype(float)
    sig["traj_human"] = nn[w, NN_HUMAN].astype(float)
    return sig


def set_thigh(length_thigh):
    """Point the retargeting geometry at one subject's thigh length.

    The module derives its constants at import, and the predictor calls through
    to it, so rebinding here is enough. Only the knee channel and the inverse
    transform depend on these.
    """
    retargeting.L_THIGH = length_thigh
    retargeting.L_HUMAN = 0.5 * (length_thigh + retargeting.L_SHANK)


def replay(sig, use_compensated_z):
    """Drive the predictor over the trial, one row per logged tick.

    Returns (rows, human, leg): the tick indices compared, and our MLP output
    and post-transform trajectory at each, both (n, 41).
    """
    p = npx.NeuralPredictor(replace(COMMON, control_sample_rate=LOG_RATE))
    st = ProsthesisState()
    hip_z = sig["hip_z_comp"] if use_compensated_z else sig["hip_z_raw"]

    rows, human, leg = [], [], []
    for k, t in enumerate(sig["time"]):
        st.hip_angle = sig["hip_angle"][k]
        st.hip_pos_x, st.hip_pos_z = sig["hip_x"][k], hip_z[k]
        p.update(st, sig["knee_desired"][k], t)

        # the network only ran while the baseline's own gate was up
        if not (sig["compute"][k] and p.ready):
            continue
        y = p._forward(p._inputs())
        rows.append(k)
        human.append(y)
        leg.append(p._human_to_leg(y))

    return np.array(rows), np.array(human), np.array(leg)


def live(ref_human):
    """Ticks where the baseline's network actually produced something.

    Its MLP sits behind a delayed enable, so for the first few ticks after the
    gate rises the logged output is still the subsystem's initial zero. The
    leg-frame trajectory is not zero there -- the transform of zero is not zero
    -- so both comparisons key off the human-frame reference.
    """
    return np.abs(ref_human).max(axis=1) > 0


def compare(ours, theirs, ok):
    """Per-tick max error over the 41 points, in degrees, plus correlation."""
    err = np.degrees(np.abs(ours[ok] - theirs[ok]).max(axis=1))
    corr = np.corrcoef(ours[ok].ravel(), theirs[ok].ravel())[0, 1]
    return err, corr


def summarize(name, ours, theirs, ok):
    err, corr = compare(ours, theirs, ok)
    dropped = int((~ok).sum())
    print(f"     {name:<22} corr {corr:.6f}   median {np.median(err):6.3f}"
          f"   p99 {np.percentile(err, 99):6.3f}   max {err.max():7.3f} deg"
          f"   ({dropped} no-reference ticks dropped)")
    return np.median(err)


def main():
    sig = load()
    print(f"walking window {sig['time'][0]:.2f}..{sig['time'][-1]:.2f} s, "
          f"{len(sig['time'])} ticks at {LOG_RATE:g} Hz")

    print("\nA) thigh length sweep, raw hip z"
          "  (median / p99 per-tick max err vs nn_traj, deg)")
    runs = {}
    for lt in THIGH_CANDIDATES:
        set_thigh(lt)
        rows, human, leg = replay(sig, use_compensated_z=False)
        ok = live(sig["traj_human"][rows])
        err, _ = compare(leg, sig["traj_leg"][rows], ok)
        runs[lt] = (np.median(err), np.percentile(err, 99), rows, human, leg, ok)
    best_len = min(runs, key=lambda k: runs[k][0])
    for lt, (med, p99, *_) in runs.items():
        print(f"     thigh {lt:.2f} m   median {med:6.3f}   p99 {p99:6.3f}"
              f"{'  <-- best' if lt == best_len else ''}")

    set_thigh(best_len)
    _, _, rows, human, leg, ok = runs[best_len]
    print(f"\nB) at thigh {best_len:.2f} m, {int(ok.sum())} ticks compared")
    summarize("mlp output (human)", human, sig["traj_human"][rows], ok)
    summarize("after transform (leg)", leg, sig["traj_leg"][rows], ok)

    print("\nC) with the load correction on hip z, which the NN does NOT use:\n"
          "      data transfer sets hip_pos_human = hip_pos, uncorrected")
    rows_r, human_r, leg_r = replay(sig, use_compensated_z=True)
    ok_r = live(sig["traj_human"][rows_r])
    summarize("mlp output (human)", human_r, sig["traj_human"][rows_r], ok_r)
    summarize("after transform (leg)", leg_r, sig["traj_leg"][rows_r], ok_r)

    plot(sig, rows, leg, rows_r, leg_r, best_len)


def plot(sig, rows, leg, rows_r, leg_r, length_thigh):
    t = sig["time"]
    ref = sig["traj_leg"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # the trajectory 50 ms out, the point the player is actually using
    j = 10
    ax[0].plot(t[rows], np.degrees(ref[rows, j]), lw=1.2, label="baseline")
    ax[0].plot(t[rows], np.degrees(leg[:, j]), lw=0.9, ls="--", label="ours")
    ax[0].set_ylabel("knee [deg]")
    ax[0].set_title(f"nn_traj sample {j} (+{j * 5} ms), thigh {length_thigh:.2f} m")
    ax[0].legend(loc="upper right")

    ax[1].plot(t[rows], np.degrees(np.abs(leg - ref[rows]).max(axis=1)), lw=0.8,
               label="load-corrected hip z")
    ax[1].plot(t[rows_r], np.degrees(np.abs(leg_r - ref[rows_r]).max(axis=1)),
               lw=0.8, label="raw hip z")
    ax[1].set_ylabel("max |err| over 41 pts [deg]")
    ax[1].set_yscale("log")
    ax[1].legend(loc="upper right")

    # one whole predicted trajectory, mid swing
    k = rows[len(rows) // 2]
    idx = np.where(rows == k)[0][0]
    ax[2].plot(np.arange(41) * 5, np.degrees(ref[k]), lw=1.4, label="baseline")
    ax[2].plot(np.arange(41) * 5, np.degrees(leg[idx]), lw=1.0, ls="--", label="ours")
    ax[2].axvline(100, color="k", lw=0.6)
    ax[2].set_xlabel("trajectory time [ms]  (100 = now)   |   top two: trial time [s]")
    ax[2].set_ylabel("knee [deg]")
    ax[2].set_title(f"one prediction at t = {t[k]:.3f} s")
    ax[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"\nwrote {PNG}")


if __name__ == "__main__":
    main()
