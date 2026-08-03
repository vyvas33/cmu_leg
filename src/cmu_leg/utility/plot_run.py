"""
Plot the CSV logs from a completed prosthesis run.

Usage:
    uv run python -m cmu_leg.utility.plot_run              # plot the latest run
    uv run python -m cmu_leg.utility.plot_run --run NAME   # plot a specific run
    uv run python -m cmu_leg.utility.plot_run --list       # list available runs

NAME can be an absolute path, a relative path, or just the directory name
(searched under output/ at the repo root).

Works on both single-process and multiprocess runs with no extra flags: every
CSV is dispatched by the suffix of its filename, so `state`, `low_prosthesis_state`
and `est_prosthesis_state` all get the state plots, `high_torque`/`torque` the
torque plots, `low_drive`/`drive` the drive plots, and every `*_idle` its own
loop-timing plot. PNG names are prefixed with the source CSV stem, so per-process
logs never clobber each other. Plots are saved into the run directory.
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


def _plot(title, t, series: dict, ylabel: str, out: Path, fname: str, prefix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, y in series.items():
        ax.plot(t, y, label=label, linewidth=1.0)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / f"{prefix}{fname}"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print("saved", png)


# ---------------------------------------------------------------------------
# per-CSV plotters
# ---------------------------------------------------------------------------

def _panel(d: dict, out: Path, prefix: str, title: str, spec: dict,
           ylabel: str, fname: str, t_key: str = "t") -> None:
    """Build one plot from a {label: (transform, column)} spec, skipping any
    column that is absent so a schema mismatch drops that series instead of
    crashing the whole run. `transform` is a per-value fn or None (identity)."""
    if t_key not in d:
        return
    t = d[t_key]
    series = {}
    for label, (fn, col) in spec.items():
        if col in d:
            series[label] = [fn(v) for v in d[col]] if fn else d[col]
    if series:
        _plot(title, t, series, ylabel, out, fname, prefix)


def _plot_state(d: dict, out: Path, prefix: str = "") -> None:
    _panel(d, out, prefix, "Knee angles",
           {"motor": (degrees, "knee_motor_angle"), "joint": (degrees, "knee_angle")},
           "angle [deg]", "knee_angles.png")
    _panel(d, out, prefix, "Knee velocities",
           {"motor": (degrees, "knee_motor_velocity"), "joint": (degrees, "knee_velocity")},
           "velocity [deg/s]", "knee_velocities.png")
    _panel(d, out, prefix, "Ankle angles",
           {"motor": (degrees, "ankle_motor_angle"), "spring": (degrees, "ankle_spring_angle"),
            "joint": (degrees, "ankle_angle")},
           "angle [deg]", "ankle_angles.png")
    _panel(d, out, prefix, "Ankle velocities",
           {"motor": (degrees, "ankle_motor_velocity"), "spring": (degrees, "ankle_spring_velocity"),
            "joint": (degrees, "ankle_velocity")},
           "velocity [deg/s]", "ankle_velocities.png")
    _panel(d, out, prefix, "Measured joint torques",
           {"knee": (None, "knee_torque_measured"), "ankle": (None, "ankle_torque_measured")},
           "torque [Nm]", "measured_torques.png")
    _panel(d, out, prefix, "Ground reaction forces",
           {"toe": (None, "toe_grf"), "heel": (None, "heel_grf"),
            "R": (None, "grf_R"), "L": (None, "grf_L")},
           "GRF", "grf.png")


def _plot_torque(d: dict, out: Path, prefix: str = "") -> None:
    _panel(d, out, prefix, "Desired torques (high-level)",
           {"knee": (None, "knee_desired"), "ankle": (None, "ankle_desired")},
           "torque [Nm]", "desired_torques.png")


def _plot_lowlevel(d: dict, out: Path, prefix: str = "") -> None:
    _panel(d, out, prefix, "Commanded torques (low-level)",
           {"knee": (None, "knee_torque_commanded"), "ankle": (None, "ankle_torque_commanded")},
           "torque [Nm]", "ll_torques_commanded.png")
    _panel(d, out, prefix, "Motor torque feedforward",
           {"knee": (None, "knee_motor_torque_ff"), "ankle": (None, "ankle_motor_torque_ff")},
           "torque [Nm]", "ll_torques_ff.png")
    _panel(d, out, prefix, "Desired joint velocities",
           {"knee": (degrees, "knee_velocity_desired"), "ankle": (degrees, "ankle_velocity_desired")},
           "velocity [deg/s]", "ll_velocity_desired.png")
    _panel(d, out, prefix, "Commanded velocities (low-level)",
           {"knee": (degrees, "knee_velocity_commanded"), "ankle": (degrees, "ankle_velocity_commanded")},
           "velocity [deg/s]", "ll_velocity_commanded.png")


def _plot_drive(d: dict, out: Path, prefix: str = "") -> None:
    _panel(d, out, prefix, "Drive velocity commands",
           {"knee": (None, "knee_velocity_counts"), "ankle": (None, "ankle_velocity_counts")},
           "counts/s", "drive_velocity_counts.png")
    _panel(d, out, prefix, "Drive torque commands",
           {"knee": (None, "knee_torque_per_mille"), "ankle": (None, "ankle_torque_per_mille")},
           "per-mille of rated torque", "drive_torque_permille.png")


def _plot_idle(d: dict, out: Path, prefix: str = "") -> None:
    """Loop-timing plot from a *_idle CSV. Uses the measured `dt` column when
    present (multiprocess) and otherwise derives it from consecutive `t`."""
    t = d["t"]
    if "dt" in d:
        dt = d["dt"]
        freq = [1.0 / x if x > 0 else 0.0 for x in dt]
        t_freq = t
    else:
        freq = [1.0 / (t[i] - t[i - 1]) for i in range(1, len(t))]
        t_freq = t[1:]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_freq, freq, linewidth=0.7)
    ax.axhline(1000, color="red", linewidth=0.8, linestyle="--", label="1 kHz")
    ax.axhline(500, color="orange", linewidth=0.8, linestyle="--", label="500 Hz")
    ax.set_title("Loop frequency", fontweight="bold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / f"{prefix}loop_frequency.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print("saved", png)


# ---------------------------------------------------------------------------
# run resolution
# ---------------------------------------------------------------------------

def _all_runs() -> list[Path]:
    out = _repo_root() / "output"
    return sorted((r for r in out.iterdir() if r.is_dir()), key=lambda p: p.name) if out.exists() else []


def _list_runs() -> None:
    runs = _all_runs()
    if not runs:
        print("No runs found in output/")
    else:
        print("Available runs:")
        for r in runs:
            print(f"  {r.name}")


def _latest_run() -> Path:
    runs = _all_runs()
    if not runs:
        raise SystemExit("No runs found in output/")
    return runs[-1]


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

# (suffix, plotter) — a CSV is handled by the first plotter whose suffix its
# filename stem ends with. This makes both single-process (`state`, `torque`,
# `drive`) and multiprocess (`low_prosthesis_state`, `high_torque`, `low_drive`,
# `*_idle`) layouts dispatch with no per-layout special-casing. Order matters:
# more specific suffixes first (e.g. `lowlevel` before `state` is moot here, but
# keep specifics ahead of generics if you add any).
_PLOTTERS = [
    ("lowlevel", _plot_lowlevel),
    ("state",    _plot_state),
    ("torque",   _plot_torque),
    ("drive",    _plot_drive),
    ("idle",     _plot_idle),
]


def _dispatch(stem: str):
    for suffix, fn in _PLOTTERS:
        if stem.endswith(suffix):
            return fn
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CSV logs from a prosthesis run (single- or multiprocess).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no arguments, plots the latest run under output/.",
    )
    parser.add_argument(
        "--run", metavar="NAME",
        help="Run directory name or path (e.g. 2026-06-30_120000_standing). "
             "Defaults to the latest run.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available runs and exit.",
    )
    args = parser.parse_args()

    if args.list:
        _list_runs()
        sys.exit(0)

    run_dir = _resolve_run(args.run) if args.run else _latest_run()
    print(f"Plotting: {run_dir}")

    plotted = 0
    for csv_path in sorted(run_dir.glob("*.csv")):
        fn = _dispatch(csv_path.stem)
        if fn is None:
            continue
        # Prefix PNGs with the source stem so per-process logs never clobber.
        prefix = f"{csv_path.stem}__"
        fn(_read_csv(csv_path), run_dir, prefix)
        plotted += 1

    if plotted == 0:
        print("  no plottable CSVs found in this run.")
    print("Done.")


if __name__ == "__main__":
    main()
