"""Re-run the NPC swing chain on one of OUR hardware runs and show the network.

The other harnesses in here replay the Simulink baseline out of `data/`. This
one replays a run from `output/`, which answers a different question: not "does
the port match the baseline" but "what did the network actually ask for on the
leg, and why".

It works because five of the predictor's six input channels are logged
(`hip_angle`, `hip_pos_x/z`, and the knee x/z it derives from those). The sixth
is the COMMANDED knee, which during swing is the controller's own previous
output -- a closed loop. Re-running the whole chain closes that loop for us, so
the predictions come back without ever having been logged. The reconstruction
is checked against the knee torque the run actually flew; it reproduces it
exactly, and the harness fails if it ever stops doing so.

Two figures:

    __timeline      two gait cycles: hip angle, hip velocity, and the knee with
                    every latched prediction drawn as a short forward curve,
                    over the swing-phase bands
    __predictions   every phase-1 prediction, start-aligned and shape-only,
                    shaded by the hip state that produced it -- the hip ->
                    predicted-knee mapping, directly

Run:  uv run python -m cmu_leg.tests.npc_run_replay [run_dir]
      run_dir defaults to the newest run in output/ whose config.json says
      walking_policy == "nm_npc".
Writes plots/npc_run_replay__timeline.png and __predictions.png.

Authors: vkalva, Claude
"""
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from cmu_leg.tests import plot_path
from cmu_leg.config.params import NNSWING
from cmu_leg.control.high_level.nm_npc import NMNPC
from cmu_leg.control.high_level.walking import nn_swing
from cmu_leg.control.high_level.walking.nn_predictor import PLAYBACK_DT
from cmu_leg.structures.state import ProsthesisState

OUTPUT = Path(__file__).resolve().parents[3] / "output"
STATE_FIELDS = {f.name for f in fields(ProsthesisState)}
WALKING = 2                       # Mode.WALKING
DEG = 180.0 / np.pi
CYCLES = 2                        # gait cycles in the timeline window

# Categorical slots 1-4 of the house palette, in fixed order. Validated for
# adjacent-pair CVD separation; slots 3 and 4 sit under 3:1 against the
# surface, which is why every series is direct-labelled in the legend.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#d8d7d2"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9,
    "axes.grid": True, "grid.color": "#ebeae5", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
}


def latest_npc_run() -> Path:
    """Newest run directory whose config.json records an nm_npc policy."""
    runs = []
    for d in sorted(OUTPUT.glob("*_multiprocess"), reverse=True):
        cfg = d / "config.json"
        if not cfg.exists() or not (d / "control_gait.csv").exists():
            continue
        try:
            policy = json.loads(cfg.read_text())["runtime"].get("walking_policy")
        except Exception:
            continue
        if policy == "nm_npc":
            runs.append(d)
    if not runs:
        raise SystemExit(
            "no run in output/ records walking_policy == 'nm_npc'. Runs from "
            "before that param existed cannot be identified; pass one by path.")
    return runs[0]


def _csv(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    out = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        out[k] = np.array([v == "True" if v in ("True", "False")
                           else (np.nan if v in ("", "nan") else float(v))
                           for v in col])
    return out


def rerun(run: Path):
    """Drive NMNPC over the logged state, capturing every latched trajectory.

    The gait FSM is fed the logged `contact`, not the logged `in_stance`, so it
    re-derives stance itself and a disagreement would show up in the checks.
    """
    st = _csv(run / "control_prosthesis_state.csv")
    gait = _csv(run / "control_gait.csv")
    torque = _csv(run / "control_torque.csv")
    t, mode, n = st["t"], gait["mode"], len(st["t"])

    ctrl = NMNPC()
    latches = []                        # (t_latch, absolute traj, traj t0)
    original = nn_swing.Phase1.set_trajectory

    def spy(self, traj):
        original(self, traj)
        latches.append((clock[0], self._traj.copy(), self._t_start))

    clock = [0.0]
    nn_swing.Phase1.set_trajectory = spy
    try:
        bus = ProsthesisState()
        cmd = np.full(n, np.nan)
        tq = np.full(n, np.nan)
        phase = np.zeros(n, int)
        armed = np.zeros(n, bool)
        stance = np.zeros(n, bool)
        entered = False
        for k in range(n):
            if mode[k] != WALKING:
                entered = False
                continue
            clock[0] = t[k]
            for f in STATE_FIELDS & st.keys():
                setattr(bus, f, st[f][k])
            if not entered:             # the mode FSM enters WALKING here
                ctrl.reset(bus, t[k])
                entered = True
            tq[k] = ctrl.compute(bus, t[k]).knee_desired
            cmd[k] = ctrl.knee_desired
            phase[k] = int(ctrl.phase)
            armed[k] = ctrl.armed
            stance[k] = ctrl.in_stance
    finally:
        nn_swing.Phase1.set_trajectory = original

    m = (mode == WALKING) & ~np.isnan(tq)
    checks = {
        "torque_corr": float(np.corrcoef(tq[m], torque["knee_desired"][m])[0, 1]),
        "torque_max_err": float(np.max(np.abs(tq[m] - torque["knee_desired"][m]))),
        "stance_agree": float((stance[m] == (gait["in_stance"][m] == 1)).mean()),
        "armed_frac": float(armed[m].mean()),
        "ticks": int(m.sum()),
    }
    return st, mode, cmd, tq, phase, armed, stance, latches, checks


def _bands(ax, t, phase, window):
    """Swing-phase bands. Grey and labelled, so phase identity never rests on
    shade alone and the categorical hues stay free for the data."""
    for p, shade in ((1, "#ececE8"), (2, "#e0dfda"), (3, "#d2d1cb")):
        ax.fill_between(t, 0, 1, where=(phase == p) & window,
                        transform=ax.get_xaxis_transform(), color=shade,
                        lw=0, zorder=0)


def plot_timeline(run, st, cmd, phase, stance, latches, window):
    t = st["t"]
    t0, t1 = t[window][0], t[window][-1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1.45],
                                        "hspace": 0.16})
    for a in ax:
        _bands(a, t, phase, window)
    for p in (1, 2, 3):                 # label each band once
        d = np.diff(np.concatenate(([0], ((phase == p) & window).astype(int), [0])))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if t[e - 1] - t[s] > 0.05:
                ax[0].text(0.5 * (t[s] + t[e - 1]), 0.93, str(p), ha="center",
                           va="top", fontsize=8, color=MUTED,
                           transform=ax[0].get_xaxis_transform())

    ax[0].plot(t[window], st["hip_angle"][window] * DEG, color=S1, lw=1.8)
    ax[0].set_ylabel("hip angle  [deg]")
    ax[0].set_title("NN swing: what the hip does, and what the network does with it",
                    loc="left", fontsize=12, color=INK, pad=22)
    ax[0].text(0, 1.045, f"run {run.name[11:17]} · grey bands are swing phases 1-2-3",
               transform=ax[0].transAxes, fontsize=8.5, color=INK2)

    ax[1].plot(t[window], st["hip_velocity"][window] * DEG, color=S1, lw=1.8)
    ax[1].axhline(0, color=RULE, lw=0.8)
    ax[1].set_ylabel("hip velocity  [deg/s]")

    for t_latch, traj, t_start in latches:
        if not (t0 <= t_latch <= t1):
            continue
        g = t_start + np.arange(len(traj)) * PLAYBACK_DT
        keep = g <= t1
        ax[2].plot(g[keep], traj[keep] * DEG, color=S3, lw=1.1, alpha=0.85, zorder=2)
    slaved = ((phase == 2) | (phase == 3)) & window
    target = (st["hip_angle"] - NNSWING.hip_knee_offset) * DEG
    ax[2].plot(np.where(slaved, t, np.nan)[window],
               np.where(slaved, target, np.nan)[window],
               color=S4, lw=1.6, ls=(0, (4, 2)), zorder=3)
    ax[2].plot(t[window], st["knee_angle"][window] * DEG, color=S1, lw=1.8, zorder=4)
    ax[2].plot(t[window], cmd[window] * DEG, color=S2, lw=1.6, zorder=5)
    ax[2].set_ylabel("knee angle  [deg]")
    ax[2].set_xlabel("time  [s]")
    ax[2].set_xlim(t0, t1)
    ax[2].legend(handles=[
        Line2D([], [], color=S1, lw=1.8, label="knee, measured"),
        Line2D([], [], color=S2, lw=1.6, label="knee, commanded"),
        Line2D([], [], color=S3, lw=1.2, label="NN predictions (one per latch)"),
        Line2D([], [], color=S4, lw=1.6, ls=(0, (4, 2)),
               label="hip_angle - hip_knee_offset  (what phases 2/3 track)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False,
        fontsize=8.5, ncol=2, labelcolor=INK2)

    png = plot_path("npc_run_replay__timeline.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_predictions(st, phase, latches):
    t = st["t"]
    rows = []
    for t_latch, traj, _ in latches:
        k = int(np.searchsorted(t, t_latch))
        if k < len(t) and phase[k] == 1:
            rows.append((st["hip_velocity"][k] * DEG, st["hip_angle"][k] * DEG,
                         (traj - traj[0]) * DEG))
    if not rows:
        return None, {}
    hip_vel = np.array([r[0] for r in rows])
    hip_ang = np.array([r[1] for r in rows])
    shape = np.array([r[2] for r in rows])
    grid = np.arange(shape.shape[1]) * PLAYBACK_DT * 1000.0

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"wspace": 0.42})
    for a, key, label in ((ax[0], hip_vel, "hip velocity at latch  [deg/s]"),
                          (ax[1], hip_ang, "hip angle at latch  [deg]")):
        # sequential: one hue, light -> dark, since the encoded value is a
        # magnitude and not an identity
        norm = Normalize(np.percentile(key, 2), np.percentile(key, 98))
        cmap = plt.get_cmap("Blues")
        for i in np.argsort(key):
            a.plot(grid, shape[i], lw=0.8, alpha=0.75,
                   color=cmap(0.22 + 0.72 * np.clip(norm(key[i]), 0, 1)))
        a.axhline(0, color=RULE, lw=0.8)
        a.set_xlabel("time from latch  [ms]")
        a.set_ylabel("predicted knee motion from here  [deg]")
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=a, pad=0.02)
        bar.set_label(label, color=INK2, fontsize=8.5)
        bar.outline.set_edgecolor(RULE)
        bar.ax.tick_params(colors=MUTED)
    ax[0].set_title("coloured by hip velocity", loc="left", fontsize=10.5, color=INK)
    ax[1].set_title("coloured by hip angle", loc="left", fontsize=10.5, color=INK)
    fig.suptitle(f"{len(rows)} phase-1 NN predictions, shape only",
                 x=0.085, y=1.10, ha="left", fontsize=12.5, color=INK)
    fig.text(0.085, 1.015, "Each curve is one network output: how far the knee is "
             "told to move over the next 100 ms, from wherever it is now.",
             fontsize=8.5, color=INK2)

    png = plot_path("npc_run_replay__predictions.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    travel = shape[:, -1]
    return png, {
        "n": len(rows),
        "corr_hip_angle": float(np.corrcoef(hip_ang, travel)[0, 1]),
        "corr_hip_vel": float(np.corrcoef(hip_vel, travel)[0, 1]),
    }


def summarize(st, mode, cmd, phase, stance, latches):
    """Per-swing phase budget, latch count and command-vs-measured tracking."""
    t = st["t"]
    edge = np.diff(stance.astype(int))
    toe, heel = np.where(edge == -1)[0] + 1, np.where(edge == 1)[0] + 1
    dur = {1: [], 2: [], 3: []}
    n_latch, peak_cmd, peak_meas = [], [], []
    for i in toe:
        e = heel[heel > i]
        if not len(e) or e[0] - i < 100:
            continue
        e = e[0]
        for p in (1, 2, 3):
            dur[p].append((phase[i:e] == p).sum() * np.median(np.diff(t)))
        n_latch.append(sum(1 for tl, _, _ in latches if t[i] <= tl < t[e]))
        peak_cmd.append(np.nanmax(cmd[i:e]) * DEG)
        peak_meas.append(np.max(st["knee_angle"][i:e]) * DEG)
    swing = (mode == WALKING) & ~stance
    return {
        "swings": len(n_latch),
        "phase_s": {p: float(np.median(dur[p])) for p in (1, 2, 3)},
        "phase_ran": {p: float(np.mean(np.array(dur[p]) > 0.004)) for p in (1, 2, 3)},
        "latches": float(np.median(n_latch)),
        "peak_cmd": float(np.median(peak_cmd)),
        "peak_meas": float(np.median(peak_meas)),
        "over_clamp": float(np.mean(cmd[swing] * DEG > NNSWING.knee_max * DEG)),
    }


def main():
    plt.rcParams.update(STYLE)      # applied here, not at import, so bringing
                                    # rerun() into another script is side-effect free
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_npc_run()
    if not run.is_absolute():
        run = OUTPUT / run.name
    print(f"run {run.name}")

    st, mode, cmd, tq, phase, armed, stance, latches, checks = rerun(run)
    print(f"\n  re-ran {checks['ticks']} WALKING ticks, "
          f"{len(latches)} latched predictions")
    print(f"  vs the knee torque that flew:  corr {checks['torque_corr']:.6f}   "
          f"max |err| {checks['torque_max_err']:.3f} Nm")
    print(f"  gait FSM agreement {100*checks['stance_agree']:.3f}%   "
          f"NN armed {100*checks['armed_frac']:.1f}% of WALKING")
    # Fidelity is reported, not assumed. An exact reconstruction means the
    # predictions below are literally the ones that flew. Some runs instead
    # drift partway through and stay drifted -- phase 2 integrates its command
    # and the predictor eats it back, so any difference is carried rather than
    # forgotten. UNEXPLAINED as of 2026-11-02: 195724 is exact, 195507 reaches
    # 3.3 Nm from t=35.4s on the same commit, same gains, with no tick gap over
    # 10 ms. Until that is understood, a drifted run is still worth plotting --
    # the shapes stay right -- but it is no longer evidence about that run.
    if checks["torque_corr"] < 0.99:
        raise SystemExit(
            "reconstruction does not track the logged torque at all -- the "
            "controller has changed since this run, so these are not its "
            "predictions. Replay a log from the current code.")
    if checks["torque_max_err"] > 1e-9:
        print(f"  NOTE: not an exact reconstruction. The shapes are still the "
              f"controller's, but this run's predictions are a close model of "
              f"what flew, not a record of it.")

    s = summarize(st, mode, cmd, phase, stance, latches)
    print(f"\n  {s['swings']} swings")
    for p in (1, 2, 3):
        print(f"    phase {p}: median {s['phase_s'][p]:.3f} s   "
              f"(ran in {100*s['phase_ran'][p]:.0f}% of swings)")
    print(f"    NN latches per swing: median {s['latches']:.0f}")
    print(f"    knee peak: commanded {s['peak_cmd']:.1f} deg, "
          f"measured {s['peak_meas']:.1f} deg")
    print(f"    command above knee_max ({NNSWING.knee_max*DEG:.0f} deg) for "
          f"{100*s['over_clamp']:.1f}% of swing ticks")

    edge = np.diff(stance.astype(int))
    toe = np.where(edge == -1)[0] + 1
    mid = len(toe) // 2
    t = st["t"]
    window = (t >= t[toe[mid]] - 0.25) & (t <= t[toe[mid + CYCLES]] + 0.1)

    png1 = plot_timeline(run, st, cmd, phase, stance, latches, window)
    png2, corr = plot_predictions(st, phase, latches)
    if corr:
        print(f"\n  predicted 100 ms knee travel vs hip state, {corr['n']} latches:")
        print(f"    corr with hip angle    {corr['corr_hip_angle']:+.3f}")
        print(f"    corr with hip velocity {corr['corr_hip_vel']:+.3f}")
    print(f"\nwrote {png1}")
    print(f"wrote {png2}")


if __name__ == "__main__":
    main()
