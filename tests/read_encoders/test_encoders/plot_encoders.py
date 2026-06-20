"""Plot the recorded encoder angles (knee + ankle) from knee.mat / ankle.mat.

For each joint, produces an overlaid plot of all its angles and a separate-panel
version. Skips the startup window (raw counts are 0 / garbage until the EtherCAT
bus is live) via START_S.

Run:  python plot_encoders.py
Outputs PNGs next to this script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).parent
START_S = 5.0  # drop the pre-bus-live startup window

# recording columns
KNEE = {"file": "knee.mat", "time": 5,
        "angles": [("motor angle", 0), ("joint angle", 1)]}
ANKLE = {"file": "ankle.mat", "time": 6,
         "angles": [("motor angle", 0), ("spring angle", 1), ("joint angle", 2)]}


def _load(spec):
    d = sio.loadmat(DATA_DIR / spec["file"])["data"]
    t = d[:, spec["time"]]
    m = t >= START_S
    return t[m], {name: np.degrees(d[m, col]) for name, col in spec["angles"]}


def plot_combined(joint, t, angles):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, y in angles.items():
        ax.plot(t, y, label=name, linewidth=1.2)
    ax.set_title(f"{joint} angles", fontweight="bold")
    ax.set_xlabel("time [s]"); ax.set_ylabel("angle [deg]")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DATA_DIR / f"{joint.lower()}_angles.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def plot_separate(joint, t, angles):
    n = len(angles)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (name, y), c in zip(axes, angles.items(), colors):
        ax.plot(t, y, color=c, linewidth=1.2)
        ax.set_title(f"{joint} {name}"); ax.set_ylabel("deg"); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    out = DATA_DIR / f"{joint.lower()}_angles_separate.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main():
    for joint, spec in (("Knee", KNEE), ("Ankle", ANKLE)):
        t, angles = _load(spec)
        print("saved", plot_combined(joint, t, angles))
        print("saved", plot_separate(joint, t, angles))


if __name__ == "__main__":
    main()
