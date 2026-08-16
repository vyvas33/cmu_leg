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

import numpy as np

_STAGES = [("t_imu", "imu.read"), ("t_buf", "buffer.latest"),
           ("t_eskf", "ESKF.update"), ("t_ctrl", "operator + pc.compute"),
           ("t_pub", "cmd_buf.publish"), ("t_log", "Recorder.append x4")]


def _load(run_dir: Path):
    idle = np.genfromtxt(run_dir / "control_idle.csv", delimiter=",", names=True)
    gait_path = run_dir / "control_gait.csv"
    gait = (np.genfromtxt(gait_path, delimiter=",", names=True)
            if gait_path.exists() else None)
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
    stance[:m] = gait["in_stance"][:m].astype(bool)
    out = [("STANDING", mode == 1),
           ("WALKING/stance", (mode == 2) & stance),
           ("WALKING/swing", (mode == 2) & ~stance)]
    return [(lbl, msk) for lbl, msk in out if msk.sum() > 50]


def report(run_dir: Path) -> None:
    idle, gait = _load(run_dir)
    cols = idle.dtype.names
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
        verdict = ("compute-bound -- fix the dominant stage" if frac < 10 else
                   "INTERFERENCE-bound -- fix the IMU stream rate / parser threads")
        print(f"    -> {verdict}\n")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for d in argv[1:]:
        report(Path(d))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
