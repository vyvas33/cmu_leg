"""
Plot the CSV logs from a completed prosthesis run.

Usage:
    uv run python -m cmu_leg.utility.plot_run <run_dir>
    uv run python -m cmu_leg.utility.plot_run          # list available runs

<run_dir> can be an absolute path, a relative path, or just the directory
name (searched under output/ at the repo root).

Plots are saved as PNGs into the same run directory.
"""

import argparse
import csv
import sys
from math import degrees
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.utility.utils import _repo_root


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> dict[str, list[float]]:
    """Return {column_name: [float, ...]} for every column in the CSV.

    Boolean fields ('True'/'False') are converted to 1.0/0.0.
    """
    _BOOL = {"True": 1.0, "False": 0.0}

    def _to_float(v: str) -> float:
        if v in _BOOL:
            return _BOOL[v]
        return float(v)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols: dict[str, list[float]] = {}
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(_to_float(v))
    return cols


def _plot(title, t, series: dict, ylabel: str, out: Path, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, y in series.items():
        ax.plot(t, y, label=label, linewidth=1.0)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / fname
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print("saved", png)


def _deg(vals: list[float]) -> list[float]:
    return [degrees(v) for v in vals]


# ---------------------------------------------------------------------------
# per-CSV plotters
# ---------------------------------------------------------------------------

def _plot_state(d: dict, out: Path) -> None:
    t = d["t"]
    _plot("Knee angles", t,
          {"motor": _deg(d["knee_motor_angle"]),
           "joint": _deg(d["knee_angle"])},
          "angle [deg]", out, "knee_angles.png")

    _plot("Knee velocities", t,
          {"motor": _deg(d["knee_motor_velocity"]),
           "joint": _deg(d["knee_velocity"])},
          "velocity [deg/s]", out, "knee_velocities.png")

    _plot("Ankle angles", t,
          {"motor": _deg(d["ankle_motor_angle"]),
           "spring": _deg(d["ankle_spring_angle"]),
           "joint": _deg(d["ankle_angle"])},
          "angle [deg]", out, "ankle_angles.png")

    _plot("Ankle velocities", t,
          {"motor": _deg(d["ankle_motor_velocity"]),
           "spring": _deg(d["ankle_spring_velocity"]),
           "joint": _deg(d["ankle_velocity"])},
          "velocity [deg/s]", out, "ankle_velocities.png")

    _plot("Measured joint torques", t,
          {"knee": d["knee_torque_measured"],
           "ankle": d["ankle_torque_measured"]},
          "torque [Nm]", out, "measured_torques.png")

    _plot("Ground reaction forces", t,
          {"toe": d["toe_grf"],
           "heel": d["heel_grf"],
           "R": d["grf_R"],
           "L": d["grf_L"]},
          "GRF", out, "grf.png")


def _plot_torque(d: dict, out: Path) -> None:
    _plot("Desired torques (high-level)", d["t"],
          {"knee": d["knee_desired"],
           "ankle": d["ankle_desired"]},
          "torque [Nm]", out, "desired_torques.png")


def _plot_lowlevel(d: dict, out: Path) -> None:
    t = d["t"]
    _plot("Commanded torques (low-level)", t,
          {"knee": d["knee_torque_commanded"],
           "ankle": d["ankle_torque_commanded"]},
          "torque [Nm]", out, "ll_torques_commanded.png")

    _plot("Motor torque feedforward", t,
          {"knee": d["knee_motor_torque_ff"],
           "ankle": d["ankle_motor_torque_ff"]},
          "torque [Nm]", out, "ll_torques_ff.png")

    _plot("Desired joint velocities", t,
          {"knee": _deg(d["knee_velocity_desired"]),
           "ankle": _deg(d["ankle_velocity_desired"])},
          "velocity [deg/s]", out, "ll_velocity_desired.png")

    _plot("Commanded velocities (low-level)", t,
          {"knee": _deg(d["knee_velocity_commanded"]),
           "ankle": _deg(d["ankle_velocity_commanded"])},
          "velocity [deg/s]", out, "ll_velocity_commanded.png")


def _plot_drive(d: dict, out: Path) -> None:
    t = d["t"]
    _plot("Drive velocity commands", t,
          {"knee": d["knee_velocity_counts"],
           "ankle": d["ankle_velocity_counts"]},
          "counts/s", out, "drive_velocity_counts.png")

    _plot("Drive torque commands", t,
          {"knee": d["knee_torque_per_mille"],
           "ankle": d["ankle_torque_per_mille"]},
          "per-mille of rated torque", out, "drive_torque_permille.png")


def _plot_loop_idle(t: list[float], out: Path) -> None:
    freq_hz = [1.0 / (t[i] - t[i - 1]) for i in range(1, len(t))]
    t_mid = [t[i] for i in range(1, len(t))]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_mid, freq_hz, linewidth=0.7)
    ax.axhline(1000, color="red", linewidth=0.8, linestyle="--", label="1 kHz target")
    ax.set_title("Control frequency", fontweight="bold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / "control_frequency.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print("saved", png)


# ---------------------------------------------------------------------------
# run resolution
# ---------------------------------------------------------------------------

def _list_runs() -> None:
    out = _repo_root() / "output"
    runs = sorted(out.iterdir()) if out.exists() else []
    if not runs:
        print("No runs found in output/")
    else:
        print("Available runs:")
        for r in runs:
            print(f"  {r.name}")


def _resolve_run(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute() and p.is_dir():
        return p
    if p.is_dir():
        return p.resolve()
    candidate = _repo_root() / "output" / arg
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"Run directory not found: {arg!r}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_PLOTTERS = {
    "state":    _plot_state,
    "torque":   _plot_torque,
    "lowlevel": _plot_lowlevel,
    "drive":    _plot_drive,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CSV logs from a prosthesis run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Omit run_dir to list available runs.",
    )
    parser.add_argument(
        "run_dir", nargs="?",
        help="Run directory name or path (e.g. 2026-06-30_120000_standing)",
    )
    args = parser.parse_args()

    if args.run_dir is None:
        _list_runs()
        sys.exit(0)

    run_dir = _resolve_run(args.run_dir)
    print(f"Plotting: {run_dir}")

    state_data = None
    for name, fn in _PLOTTERS.items():
        csv_path = run_dir / f"{name}.csv"
        if not csv_path.exists():
            print(f"  skipping {name}.csv (not found)")
            continue
        d = _read_csv(csv_path)
        if name == "state":
            state_data = d
        fn(d, run_dir)

    if state_data is not None:
        _plot_loop_idle(state_data["t"], run_dir)

    print("Done.")


if __name__ == "__main__":
    main()
