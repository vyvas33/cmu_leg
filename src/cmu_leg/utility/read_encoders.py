"""
Read and plot the encoder-derived prosthesis state.

Usage:
uv run python -m cmu_leg.utility.read_encoders


vkalva, June 2026
"""

import time
from math import degrees
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.estimation.state_estimator import StateEstimator
from cmu_leg.hardware.ethercat_io import EtherCATInterface
from cmu_leg.types.sensors import IMUFrame
from cmu_leg.utility.utils import make_run_dir
from cmu_leg.utility.recorder import Recorder
from cmu_leg.control.utils import Lowpass
from cmu_leg.config.params import STANDING, COMMON

DURATION = 20.0  # seconds

def _plot(title, t, series, ylabel, out, fname):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, y in series.items():
        ax.plot(t, y, label=name, linewidth=1.0)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("time [s]"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / fname
    fig.savefig(png, dpi=110); plt.close(fig)
    print("saved", png)

def _plot_idle(t, idle_ms, out):
    """Loop slack before each 1 kHz tick, in ms."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, idle_ms, linewidth=0.7)
    ax.set_title("Loop idle time", fontweight="bold")
    ax.set_xlabel("loop time [s]"); ax.set_ylabel("idle [ms]")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", label="overrun")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / "loop_idle.png"
    fig.savefig(png, dpi=110); plt.close(fig)
    print("saved", png)

def main():
    ec = EtherCATInterface("eth0")
    ec.connect()
    se = StateEstimator()
    print(f"Recording for {DURATION} seconds")

    state_rec = Recorder("state")    # full ProsthesisState per tick -> state.csv
    frame_rec = Recorder("frame")    # raw EtherCAT PDO per tick   -> frame.csv
    idle_log = []                    # per-iteration loop slack (ms), timing diagnosis

    knee_vel_lp = Lowpass(cutoff_freq=STANDING.knee_vel_lowpass_cutoff,
                          sample_rate=COMMON.sample_rate, order=STANDING.vel_lowpass_order)
    ankle_vel_lp = Lowpass(cutoff_freq=STANDING.ankle_vel_lowpass_cutoff,
                           sample_rate=COMMON.sample_rate, order=STANDING.vel_lowpass_order)
    knee_vel_filt, ankle_vel_filt = [], []

    period = 0.001   # target loop period (s) -> ~1 kHz
    t0 = time.perf_counter()
    clk = t0
    t_wall_prev = t0
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        while (t := time.perf_counter() - t0) < DURATION:
            clk += period
            now = time.perf_counter()
            dt, t_wall_prev = now - t_wall_prev, now
            raw = ec.read()
            s = se.get_state(raw, IMUFrame(), t, dt)   # encoder-only; empty IMU frame
            state_rec.append(t, s)
            frame_rec.append(t, raw)               # raw counts + Medulla timestamps
            knee_vel_filt.append(degrees(knee_vel_lp.compute(s.knee_velocity)))
            ankle_vel_filt.append(degrees(ankle_vel_lp.compute(s.ankle_velocity)))

            # sleep only the time left in this period
            idle = clk - time.perf_counter()
            idle_log.append(idle * 1000.0)   # ms
            if idle > 0:
                time.sleep(idle)
    except KeyboardInterrupt:
        print("terminated early")
    finally:
        gc.enable()
        ec.close()

    out = make_run_dir("read_encoders")
    state_rec.save(out)
    frame_rec.save(out)

    # Build plot series straight from the recorded columns 
    t = state_rec.column("t")
    deg = lambda name: [degrees(v) for v in state_rec.column(name)]
    _plot("Knee angles", t,
          {"motor": deg("knee_motor_angle"), "joint": deg("knee_angle")},
          "angle [deg]", out, "knee_angles.png")
    _plot("Knee velocities", t,
          {"motor": deg("knee_motor_velocity"), "joint": deg("knee_velocity"),
           "joint filt": knee_vel_filt},
          "velocity [deg/s]", out, "knee_velocities.png")
    _plot("Ankle angles", t,
          {"motor": deg("ankle_motor_angle"), "spring": deg("ankle_spring_angle"),
           "joint": deg("ankle_angle")},
          "angle [deg]", out, "ankle_angles.png")
    _plot("Ankle velocities", t,
          {"motor": deg("ankle_motor_velocity"), "spring": deg("ankle_spring_velocity"),
           "joint": deg("ankle_velocity"), "joint filt": ankle_vel_filt},
          "velocity [deg/s]", out, "ankle_velocities.png")
    _plot("Measured joint torques", t,
          {"knee": state_rec.column("knee_torque_measured"),
           "ankle": state_rec.column("ankle_torque_measured")},
          "torque [Nm]", out, "torques.png")
    _plot("Hall GRF", t,
          {"toe": state_rec.column("toe_grf"),
           "heel": state_rec.column("heel_grf")},
          "GRF [counts]", out, "hall_grf.png")
    _plot_idle(t, idle_log, out)

if __name__ == "__main__":
    main()
