"""
Read and plot the encoder-derived prosthesis state.

Usage:
uv run python -m cmu_leg.utility.read_encoders
sudo .venv/bin/python -m cmu_leg.utility.read_encoders

"""

import time
from math import degrees

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

DURATION = 40.0  # seconds

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

    rec = Recorder("state")          # full ProsthesisState per tick -> state.csv
    t_log = []
    idle_log = []                    # per-iteration loop slack (ms), timing diagnosis
    knee_ang = {"motor": [], "joint": []}
    knee_vel = {"motor": [], "joint": [], "joint filt": []}
    ankle_ang = {"motor": [], "spring": [], "joint": []}
    ankle_vel = {"motor": [], "spring": [], "joint": [], "joint filt": []}
    torque = {"knee": [], "ankle": []}

    # Mirror the standing controller's velocity lowpass (STANDING cutoffs/order)
    # so the "joint filt" trace is exactly the velocity the controller acts on.
    knee_vel_lp = Lowpass(cutoff_freq=STANDING.knee_vel_lowpass_cutoff,
                          sample_rate=COMMON.sample_rate, order=STANDING.vel_lowpass_order)
    ankle_vel_lp = Lowpass(cutoff_freq=STANDING.ankle_vel_lowpass_cutoff,
                           sample_rate=COMMON.sample_rate, order=STANDING.vel_lowpass_order)

    period = 0.001   # target loop period (s) -> ~1 kHz
    t0 = time.perf_counter()
    clk = t0
    try:
        while (t := time.perf_counter() - t0) < DURATION:
            clk += period
            s = se.get_state(ec.read(), IMUFrame(), t)   # encoder-only; empty IMU frame
            rec.append(t, s)
            t_log.append(t)

            knee_ang["motor"].append(degrees(s.knee_motor_angle))
            knee_ang["joint"].append(degrees(s.knee_angle))
            knee_vel["motor"].append(degrees(s.knee_motor_velocity))
            knee_vel["joint"].append(degrees(s.knee_velocity))
            knee_vel["joint filt"].append(degrees(knee_vel_lp.compute(s.knee_velocity)))

            ankle_ang["motor"].append(degrees(s.ankle_motor_angle))
            ankle_ang["spring"].append(degrees(s.ankle_spring_angle))
            ankle_ang["joint"].append(degrees(s.ankle_angle))
            ankle_vel["motor"].append(degrees(s.ankle_motor_velocity))
            ankle_vel["spring"].append(degrees(s.ankle_spring_velocity))
            ankle_vel["joint"].append(degrees(s.ankle_velocity))
            ankle_vel["joint filt"].append(degrees(ankle_vel_lp.compute(s.ankle_velocity)))

            torque["knee"].append(s.knee_torque_measured)
            torque["ankle"].append(s.ankle_torque_measured)

            # sleep only the time left in this period
            idle = clk - time.perf_counter()
            idle_log.append(idle * 1000.0)   # ms
            if idle > 0:
                time.sleep(idle)
    except KeyboardInterrupt:
        print("terminated early")
    finally:
        ec.close()

    out = make_run_dir("read_encoders")
    rec.save(out)
    _plot("Knee angles", t_log, knee_ang, "angle [deg]", out, "knee_angles.png")
    _plot("Knee velocities", t_log, knee_vel, "velocity [deg/s]", out, "knee_velocities.png")
    _plot("Ankle angles", t_log, ankle_ang, "angle [deg]", out, "ankle_angles.png")
    _plot("Ankle velocities", t_log, ankle_vel, "velocity [deg/s]", out, "ankle_velocities.png")
    _plot("Measured joint torques", t_log, torque, "torque [Nm]", out, "torques.png")
    _plot_idle(t_log, idle_log, out)

if __name__ == "__main__":
    main()
