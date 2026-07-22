"""Animate the leg in the sagittal plane from a completed run.

Usage:
    uv run python -m cmu_leg.utility.visualise_leg <run_dir>
    uv run python -m cmu_leg.utility.visualise_leg <run_dir> --fps 20
    uv run python -m cmu_leg.utility.visualise_leg              # list runs

Writes leg.gif into the run directory. The hip is pinned at the origin, so this
shows the leg's shape over time, not where it is in the world.
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from cmu_leg.estimation.leg_kinematics import leg_points
from cmu_leg.utility.plot_run import _read_csv, _resolve_run, _list_runs


def _frames(run_dir, fps: int):
    d = _read_csv(run_dir / "state.csv")
    t = np.asarray(d["t"])
    hip = np.asarray(d["hip_angle"])
    knee = np.asarray(d["knee_angle"])
    ankle = np.asarray(d["ankle_angle"])

    stride = max(1, int(round((1.0 / fps) / np.median(np.diff(t)))))
    return t[::stride], hip[::stride], knee[::stride], ankle[::stride]


def animate(run_dir, fps: int = 30) -> None:
    t, hip, knee, ankle = _frames(run_dir, fps)
    pts = [leg_points(h, k, a) for h, k, a in zip(hip, knee, ankle)]

    xs = np.concatenate([[p.hip[0], p.knee[0], p.ankle[0], p.toe[0], p.heel[0]] for p in pts])
    zs = np.concatenate([[p.hip[1], p.knee[1], p.ankle[1], p.toe[1], p.heel[1]] for p in pts])
    pad = 0.12

    fig, ax = plt.subplots(figsize=(5.5, 7.4))
    fig.subplots_adjust(bottom=0.14)
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(zs.min() - pad, zs.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (+forward)")
    ax.set_ylabel("z [m]  (+up)")
    ax.grid(alpha=0.3)

    thigh_ln, = ax.plot([], [], lw=4, color="#3b6ea5", solid_capstyle="round")
    shank_ln, = ax.plot([], [], lw=4, color="#c26b3e", solid_capstyle="round")
    foot_ln, = ax.plot([], [], lw=4, color="#4c8b5b", solid_capstyle="round")
    joints, = ax.plot([], [], "o", ms=8, color="#222", zorder=3)
    trail, = ax.plot([], [], lw=1.0, color="#c26b3e", alpha=0.35)
    readout = fig.text(0.5, 0.045, "", ha="center", family="monospace", fontsize=9)

    def update(i):
        p = pts[i]
        thigh_ln.set_data([p.hip[0], p.knee[0]], [p.hip[1], p.knee[1]])
        shank_ln.set_data([p.knee[0], p.ankle[0]], [p.knee[1], p.ankle[1]])
        foot_ln.set_data([p.heel[0], p.ankle[0], p.toe[0]],
                         [p.heel[1], p.ankle[1], p.toe[1]])
        joints.set_data([p.hip[0], p.knee[0], p.ankle[0]],
                        [p.hip[1], p.knee[1], p.ankle[1]])
        trail.set_data([q.ankle[0] for q in pts[:i + 1]],
                       [q.ankle[1] for q in pts[:i + 1]])
        readout.set_text(
            f"t {t[i]:6.2f} s    hip {np.degrees(hip[i]):6.2f}    "
            f"knee {np.degrees(knee[i]):6.2f}    ankle {np.degrees(ankle[i]):6.2f}   (deg)")
        return thigh_ln, shank_ln, foot_ln, joints, trail, readout

    ax.set_title(run_dir.name, fontweight="bold", fontsize=10)
    anim = FuncAnimation(fig, update, frames=len(pts), interval=1000 / fps, blit=False)

    out = run_dir / "leg.gif"
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out}  ({len(pts)} frames @ {fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.run_dir is None:
        _list_runs()
        sys.exit(0)

    run_dir = _resolve_run(args.run_dir)
    print(f"Animating: {run_dir}")
    animate(run_dir, args.fps)


if __name__ == "__main__":
    main()
