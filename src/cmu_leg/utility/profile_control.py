"""Read a run's control_idle.csv and say where the 500 Hz tick went.

    uv run python -m cmu_leg.utility.profile_control output/<run_dir>

Splits the tick by stage and by gait phase, and -- the point of it -- separates
work from interference. Each stage is reported in WALL time; the run-level
`cpu_body` column says how much of that wall time the control thread actually
spent on a CPU. The gap is time it was descheduled or waiting on the GIL.

Reading the verdict line:

  * stolen time small (< ~10% of the body)
      The loop is genuinely compute-bound. Whichever stage dominates `t_*` is
      the thing to fix -- decimate it, relocate it, or make it cheaper.

  * stolen time large
      The stages are not slow, they are being interrupted. The two IMU parser
      threads share this core and this interpreter (params.IMU.stream_rate_hz
      packets/s each), and they inherit the control process's SCHED_FIFO
      priority because pin_to_core runs before imu.connect(). Lowering the
      stream rate or moving the IMUs to their own process is the fix, and
      making the ESKF cheaper would buy comparatively little.

Requires the `t_*` / `cpu_body` columns added to runtime.control.LoopStats;
older runs only have `idle` / `dt` and are reported as such.
"""

import sys
from pathlib import Path

import csv

import numpy as np

_STAGES = [("t_imu", "imu.read"), ("t_buf", "buffer.latest"),
           ("t_eskf", "ESKF.update"), ("t_ctrl", "operator + pc.compute"),
           ("t_pub", "cmd_buf.publish"), ("t_tail", "arm + stop_event.is_set"),
           ("t_log", "Recorder.append x4")]


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    """{column: float array} for a Recorder CSV.

    Recorder writes Python bools through csv.writer, so boolean columns land as
    the literal strings 'True'/'False' -- and they MUST be decoded here.
    np.genfromtxt types every column float64, parses those strings as nan, and
    nan is truthy, so `col.astype(bool)` silently returns True for every row.
    That is not hypothetical: it made a run whose gait FSM was cycling normally
    read as 100% stance with zero swing ticks. utility.plot_run._read_csv has
    always done this decode, which is why the plots were right.
    """
    bools = {"True": 1.0, "False": 0.0}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols: dict[str, list[float]] = {}
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(bools.get(v, np.nan)
                                              if v in bools else float(v))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _load(run_dir: Path):
    idle = _read_csv(run_dir / "control_idle.csv")
    gait_path = run_dir / "control_gait.csv"
    gait = _read_csv(gait_path) if gait_path.exists() else None
    return idle, gait


def _phases(idle, gait):
    """(label, mask) per gait phase, or a single 'whole run' if no gait log."""
    n = len(idle["t"])
    if gait is None:
        return [("whole run", np.ones(n, bool))]
    m = min(n, len(gait["t"]))
    mode = np.full(n, -1)
    stance = np.zeros(n, bool)
    mode[:m] = gait["mode"][:m].astype(int)
    stance[:m] = gait["in_stance"][:m] > 0.5   # decoded 1.0/0.0, see _read_csv
    out = [("STANDING", mode == 1),
           ("WALKING/stance", (mode == 2) & stance),
           ("WALKING/swing", (mode == 2) & ~stance)]
    return [(lbl, msk) for lbl, msk in out if msk.sum() > 50]


def _nominal_period(run_dir: Path, idle) -> float:
    """The period the loop is TRYING to hold, from the run's own config.json.
    Falls back to the median measured dt when there is no config (older runs)."""
    cfg = run_dir / "config.json"
    if cfg.exists():
        import json
        try:
            rate = json.loads(cfg.read_text())["common"]["control_sample_rate"]
            return 1.0 / float(rate)
        except (KeyError, ValueError, TypeError):
            pass
    return float(np.median(idle["dt"]))


def report(run_dir: Path) -> None:
    idle, gait = _load(run_dir)
    cols = set(idle)
    nominal = _nominal_period(run_dir, idle)
    print(f"\n{run_dir.name}   {len(idle['t'])} ticks, "
          f"{idle['t'][-1] - idle['t'][0]:.1f} s\n")

    if "t_body" not in cols:
        dt = idle["dt"]
        print("  (no per-stage columns -- pre-instrumentation run)")
        print(f"  median dt {np.median(dt) * 1e3:.3f} ms -> {1 / np.median(dt):.1f} Hz")
        return

    for label, m in _phases(idle, gait):
        body = idle["t_body"][m]
        cpu = idle["cpu_body"][m]
        dt = idle["dt"][m]
        stolen = body - cpu
        print(f"  {label}   n={m.sum()}   "
              f"tick {np.median(dt) * 1e3:.3f} ms ({1 / np.median(dt):.0f} Hz)")
        print(f"    {'stage':24s} {'median':>9} {'p95':>9} {'share':>7}")
        tot = np.median(body)
        for key, name in _STAGES:
            if key not in cols:
                continue
            v = idle[key][m]
            print(f"    {name:24s} {np.median(v) * 1e6:8.1f}us "
                  f"{np.percentile(v, 95) * 1e6:8.1f}us "
                  f"{np.median(v) / tot * 100:6.1f}%")
        print(f"    {'-' * 52}")
        print(f"    {'body (wall)':24s} {np.median(body) * 1e6:8.1f}us")
        print(f"    {'body (this thread CPU)':24s} {np.median(cpu) * 1e6:8.1f}us")
        frac = np.median(stolen) / tot * 100 if tot else 0.0
        print(f"    {'STOLEN (wall - cpu)':24s} {np.median(stolen) * 1e6:8.1f}us "
              f"{np.percentile(stolen, 95) * 1e6:8.1f}us {frac:6.1f}%")
        # Rank the two terms against the PERIOD, not against each other: what
        # matters is whether CPU alone already overruns. A loop can be both
        # compute-heavy and contended, and calling it "interference-bound" just
        # because stolen time cleared some fixed percentage is wrong -- an
        # earlier version of this script did exactly that and mislabelled a run
        # whose ESKF was 85% of the body.
        period = np.median(dt)
        top = max(((k, np.median(idle[k][m])) for k, _ in _STAGES if k in cols),
                  key=lambda kv: kv[1])
        top_name = dict(_STAGES)[top[0]]
        # Per-tick work vs the period. NOT `idle < 0`: runtime.control advances
        # its deadline with a bare `clock += dt` accumulator, so `idle` records
        # accumulated schedule DEBT (a stall during ESKF init buries it, and it
        # only recovers while work is strictly under budget) rather than what
        # this tick cost. An earlier version of this script reported idle < 0 as
        # the overrun rate and made a loop with 155 us of real margin look like
        # it was missing two thirds of its deadlines.
        work = idle["t_body"][m] + idle["t_log"][m]
        over = work > nominal
        print(f"    CPU alone is {np.median(cpu) / period * 100:.0f}% of the "
              f"{period * 1e3:.3f} ms period; stolen adds {frac:.0f}% of body")
        print(f"    work/tick (body+log) vs {nominal * 1e3:.3f} ms budget: "
              f"p50 {np.median(work) * 1e6:.0f}us  p95 "
              f"{np.percentile(work, 95) * 1e6:.0f}us  -> OVER budget on "
              f"{over.mean() * 100:.1f}% of ticks")
        debt = idle["idle"][m]
        print(f"    schedule debt (idle, accumulator -- see note): p50 "
              f"{np.median(debt) * 1e6:+.0f}us  p5 "
              f"{np.percentile(debt, 5) * 1e6:+.0f}us")
        if np.median(cpu) > 0.8 * period:
            print(f"    -> COMPUTE-bound: '{top_name}' is {top[1] / tot * 100:.0f}% "
                  f"of the body. Contention is secondary.")
        elif frac >= 15:
            print("    -> INTERFERENCE-bound: CPU fits the period; fix the IMU "
                  "stream rate / parser threads.")
        else:
            print(f"    -> fits, with margin. Dominant stage: '{top_name}'.")
        print()


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for d in argv[1:]:
        report(Path(d))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
