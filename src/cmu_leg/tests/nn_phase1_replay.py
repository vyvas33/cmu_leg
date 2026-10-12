"""Replay Phase1 against the baseline's logged phase 1 output.

The predictor is kept out of it: the trajectories come from the logged
`nn_traj`, resampled with our own spline matrix, so a disagreement here is the
clock, the swap timing or the anchoring.

Three references, all from `nntorque`:
    col 9  nn_prediction_time   the baseline's playback clock
    col 5  condition_delay      1 on the tick it swapped trajectories
    col 6  nn_knee_angle        the phase 1 command

Phase 1 anchors onto its own previous output, so once a swing starts the
command is open loop. Only the first anchor comes from the log.

Run:  uv run python -m cmu_leg.tests.nn_phase1_replay
Writes nn_phase1_replay.png next to this file.

Authors: vkalva, Claude
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.config.params import COMMON
from cmu_leg.control.high_level.walking.nn_swing import Phase1
from cmu_leg.control.high_level.walking.nn_predictor import SPLINE_FUTURE

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = HERE / "nn_phase1_replay.png"

NN_TRAJ = slice(2, 43)
NT = dict(knee_angle=6, swapped=5, clock=9, knee_desired=23, phase1=2, time=25)

LOG_RATE = 1000.0
WALK_START_S = 16.5


def load():
    sd = sio.loadmat(str(MAT), squeeze_me=True, struct_as_record=False)["scope_data"]
    nn, nt = sd.nn.data, sd.nntorque.data
    w = nt[:, NT["time"]] >= WALK_START_S
    sig = {k: nt[w, c].astype(float) for k, c in NT.items()}
    sig["traj"] = nn[w, NN_TRAJ].astype(float)
    return sig


def swings(phase1):
    """(start, end) index pairs for each phase 1 episode."""
    d = np.diff(np.r_[0, phase1 != 0, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def replay_swing(sig, s, e):
    """Run Phase1 over one episode. Returns our command and clock per tick."""
    p = Phase1(common=replace(COMMON, control_sample_rate=LOG_RATE))
    t = sig["time"]
    p.reset(sig["knee_desired"][s - 1], t[s])

    cmd = np.empty(e - s)
    clock = np.empty(e - s)
    swapped = np.zeros(e - s, bool)
    for i, k in enumerate(range(s, e)):
        if p.needs_trajectory(t[k]):
            p.set_trajectory(SPLINE_FUTURE @ sig["traj"][k])
            swapped[i] = True
        clock[i] = t[k] - p._t_start
        cmd[i] = p.compute(t[k])
    return cmd, clock, swapped


def main():
    sig = load()
    eps = swings(sig["phase1"])
    print(f"{len(eps)} phase 1 episodes in the walking window\n")
    print("   start      ticks   swaps(ours/base)   clock max|err|   "
          "command max|err|  rms")

    out = []
    for s, e in eps:
        cmd, clock, swapped = replay_swing(sig, s, e)
        ref = sig["knee_angle"][s:e]
        ref_clock = sig["clock"][s:e]
        ref_swap = sig["swapped"][s:e] != 0
        err = np.degrees(np.abs(cmd - ref))
        print(f"   {sig['time'][s]:7.3f}  {e - s:6d}   {swapped.sum():4d}/"
              f"{ref_swap.sum():<4d}      {1e3 * np.abs(clock - ref_clock).max():7.3f} ms"
              f"     {err.max():8.4f} deg  {err.std():7.4f}")
        out.append((s, e, cmd, clock, swapped, err))

    all_err = np.concatenate([o[5] for o in out])
    print(f"\n   all {len(all_err)} ticks: max {all_err.max():.4f} deg, "
          f"rms {all_err.std():.4f} deg")
    plot(sig, out)


def plot(sig, out):
    t = sig["time"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for i, (s, e, cmd, clock, swapped, err) in enumerate(out):
        lbl = dict(label="baseline") if i == 0 else {}
        ax[0].plot(t[s:e], np.degrees(sig["knee_angle"][s:e]), "k", lw=1.3, **lbl)
        lbl = dict(label="ours") if i == 0 else {}
        ax[0].plot(t[s:e], np.degrees(cmd), "C1--", lw=1.0, **lbl)
        ax[1].plot(t[s:e], err, "C1", lw=0.9)
        ax[1].plot(t[s:e][swapped], err[swapped], "C0.", ms=5)
    ax[0].set_ylabel("phase 1 knee command [deg]")
    ax[0].legend(loc="upper left")
    ax[1].set_ylabel("|error| [deg]")
    ax[1].set_xlabel("trial time [s]   (dots = trajectory swaps)")
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"\nwrote {PNG}")


if __name__ == "__main__":
    main()
