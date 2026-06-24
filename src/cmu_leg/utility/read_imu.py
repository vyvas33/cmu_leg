"""
A script to read and plot thigh and foot IMU sensor streams
"""

import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.hardware.imu_io import IMUInterface, discover_imu_bauds
from cmu_leg.utility.utils import make_run_dir

DURATION = 10.0  # seconds


def _plot(name, t, acc, gyro, out):
    """One figure per IMU, dumps the same series to CSV."""
    ax_x, ax_y, ax_z = zip(*acc) if acc else ([], [], [])
    gx, gy, gz = zip(*gyro) if gyro else ([], [], [])

    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for series, lbl in zip((ax_x, ax_y, ax_z), ("x", "y", "z")):
        ax_a.plot(t, series, label=lbl, linewidth=1.0)
    ax_a.set_title(f"{name} IMU accelerometer", fontweight="bold")
    ax_a.set_ylabel("accel [m/s²]")
    ax_a.legend(); ax_a.grid(alpha=0.3)

    for series, lbl in zip((gx, gy, gz), ("x", "y", "z")):
        ax_g.plot(t, series, label=lbl, linewidth=1.0)
    ax_g.set_title(f"{name} IMU gyroscope", fontweight="bold")
    ax_g.set_xlabel("time [s]"); ax_g.set_ylabel("gyro [rad/s]")
    ax_g.legend(); ax_g.grid(alpha=0.3)

    fig.tight_layout()
    png = out / f"{name.lower()}_imu.png"
    fig.savefig(png, dpi=110); plt.close(fig)

    csv_path = out / f"{name.lower()}_imu.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time [s]", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        w.writerows(zip(t, ax_x, ax_y, ax_z, gx, gy, gz))
    print("saved", png, "and", csv_path)


def _plot_idle(t, idle_ms, out):
    """Loop slack before each 1 kHz tick, in ms. Dips toward/below zero mean the
    loop overran its budget — exactly where the single-slot mailbox drops frames."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, idle_ms, linewidth=0.7)
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1.0, label="overrun threshold")
    ax.set_title("Loop idle time", fontweight="bold")
    ax.set_xlabel("loop time [s]"); ax.set_ylabel("idle [ms]")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / "loop_idle.png"
    fig.savefig(png, dpi=110); plt.close(fig)
    print("saved", png)


def main():
    thigh_baud, foot_baud = discover_imu_bauds()
    imu = IMUInterface(thigh_baud=thigh_baud, foot_baud=foot_baud)
    imu.connect()
    print(f"Recording IMU data for {DURATION} seconds")

    # per-IMU buffers; the two streams are async, so each is logged on its own clock
    t_thigh, thigh_acc, thigh_gyro = [], [], []
    t_foot, foot_acc, foot_gyro = [], [], []
    t_loop, idle_log = [], []   # per-iteration loop slack (ms), for timing diagnosis

    period = 0.001 # target loop period 1kHz (oversamples the IMU stream)
    t0 = time.perf_counter()
    clk = t0
    try:
        while (t := time.perf_counter() - t0) < DURATION:
            frame = imu.read()
            # read() repeats the last frame between sensor updates; 
            if frame.thigh_fresh:
                t_thigh.append(frame.thigh_time)
                thigh_acc.append(frame.thigh_acc)
                thigh_gyro.append(frame.thigh_gyro)
            if frame.foot_fresh:
                t_foot.append(frame.foot_time)
                foot_acc.append(frame.foot_acc)
                foot_gyro.append(frame.foot_gyro)

            clk += period
            idle = clk - time.perf_counter()
            t_loop.append(t)
            idle_log.append(idle * 1000.0)   # ms
            if idle > 0:
                time.sleep(idle)
    except KeyboardInterrupt:
        print("terminated early")
    finally:
        imu.close()

    out = make_run_dir("read_imu")
    _plot("Thigh", t_thigh, thigh_acc, thigh_gyro, out)
    _plot("Foot", t_foot, foot_acc, foot_gyro, out)
    _plot_idle(t_loop, idle_log, out)

if __name__ == "__main__":
    main()
