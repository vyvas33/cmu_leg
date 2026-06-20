"""Online read_encoders: bring up the live EtherCAT bus, record, and plot.

The hardware counterpart to test_encoders_offline.py (which replays .mat files).
Brings the bus to OP via EtherCATInterface, logs engineering-unit angles for
DURATION_S while you move the leg, and saves overlaid angle plots in the same
style as plot_encoders.py.

Run:  sudo /home/cmuleg/cmu_leg/.venv/bin/python \
          tests/read_encoders/test_encoders/read_encoders_online.py
Stop early with Ctrl+C (never ^Z — it leaves the master holding the NIC).
"""
import sys
import time
from math import degrees
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# put tests/ on the path so `read_encoders` (the package) resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from read_encoders import EtherCATInterface, read_encoders  # noqa: E402

DURATION_S = 30.0
OUT_DIR = Path(__file__).parent


def _plot(joint, t, angles):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, y in angles.items():
        ax.plot(t, y, label=name, linewidth=1.2)
    ax.set_title(f"{joint} angles (live)", fontweight="bold")
    ax.set_xlabel("time [s]"); ax.set_ylabel("angle [deg]")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"{joint.lower()}_angles_live.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print("saved", out)


def main():
    ec = EtherCATInterface("eth0")
    ec.connect()
    print(f"recording {DURATION_S:.0f} s - move the knee and ankle")

    t_log = []
    knee = {"motor angle": [], "joint angle": []}
    ankle = {"motor angle": [], "spring angle": [], "joint angle": []}

    t0 = time.perf_counter()
    try:
        while (t := time.perf_counter() - t0) < DURATION_S:
            r = read_encoders(ec.read())
            t_log.append(t)
            knee["motor angle"].append(degrees(r.knee_motor_angle))
            knee["joint angle"].append(degrees(r.knee_angle))
            ankle["motor angle"].append(degrees(r.ankle_motor_angle))
            ankle["spring angle"].append(degrees(r.ankle_spring_angle))
            ankle["joint angle"].append(degrees(r.ankle_angle))
            time.sleep(0.005)   # ~200 Hz, readable rate (not the real loop rate)
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        ec.close()

    _plot("Knee", t_log, knee)
    _plot("Ankle", t_log, ankle)


if __name__ == "__main__":
    main()
