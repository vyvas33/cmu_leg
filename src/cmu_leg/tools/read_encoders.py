import csv
import time
from dataclasses import dataclass
from math import degrees, floor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.types.sensors import RawEtherCATFrame
from cmu_leg.config.params import ENCODERS, ENCODER_CORRECTION, KNEE, ANKLE
from cmu_leg.hardware.ethercat_io import EtherCATInterface
from cmu_leg.tools.paths import make_run_dir

DURATION = 10.0  # seconds

@dataclass(slots=True)
class SensorReadings:
    """Engineering-unit outputs of read_encoders (radians, Nm)."""
    knee_motor_angle: float = 0.0
    knee_angle: float = 0.0
    knee_torque: float = 0.0
    ankle_motor_angle: float = 0.0
    ankle_spring_angle: float = 0.0
    ankle_angle: float = 0.0
    ankle_torque: float = 0.0

def _polyval(coeffs: tuple[float, ...], x: float) -> float:
    """Horner evaluation, highest-power-first (matches MATLAB/numpy polyval)."""
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc

def _wrap_counts(counts: int, total: int) -> int:
    """Wrap absolute counts into [-total/2, total/2).
    """
    return counts - total * floor((counts + total / 2) / total)

def _plot(joint, t, angles, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, y in angles.items():
        ax.plot(t, y, label=name, linewidth=1.2)
    ax.set_title(f"{joint} angles", fontweight="bold")
    ax.set_xlabel("time [s]"); ax.set_ylabel("angle [deg]")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out / f"{joint}_angles.png"
    fig.savefig(png, dpi=110); plt.close(fig)

    # dump the same series to CSV: time + one column per angle (deg)
    names = list(angles)
    csv_path = out / f"{joint}_angles.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time [s]", *names])
        w.writerows(zip(t, *(angles[n] for n in names)))
    print("saved", png, "and", csv_path)

def _convert_to_readings(frame: RawEtherCATFrame) -> SensorReadings:
    knee_motor_angle = KNEE.motor_abs_counts_to_rad * (frame.knee_motor_counts - ENCODERS.knee_motor_abs_counts_offset)

    # knee joint angle
    kc = frame.knee_joint_counts - ENCODERS.knee_abs_counts_offset
    knee_lin = ENCODERS.knee_abs_counts_to_rad * kc
    knee_norm = (kc - ENCODER_CORRECTION.knee_counts_mean) / ENCODER_CORRECTION.knee_counts_std
    knee_err = _polyval(ENCODER_CORRECTION.knee_err_poly, knee_norm) * ENCODER_CORRECTION.knee_err_std + ENCODER_CORRECTION.knee_err_mean
    knee_angle = knee_lin + knee_err

    # knee torque
    knee_torque = KNEE.stiffness * (knee_motor_angle - knee_angle)

    ankle_motor_angle = ANKLE.motor_abs_counts_to_rad * (frame.ankle_motor_counts - ENCODERS.ankle_motor_abs_counts_offset)

    awc = _wrap_counts(frame.ankle_joint_counts, ENCODER_CORRECTION.ankle_total_counts)
    ac = awc - ENCODERS.ankle_abs_counts_offset
    ankle_angle = ENCODERS.ankle_abs_counts_to_rad * ac

    # ankle spring angle
    spring_lin = _polyval(ENCODER_CORRECTION.ankle_spring_counts_to_rad_poly, ac)
    spring_norm = (ac - ENCODER_CORRECTION.ankle_spring_counts_mean) / ENCODER_CORRECTION.ankle_spring_counts_std
    spring_err = (
        _polyval(ENCODER_CORRECTION.ankle_spring_err_poly, spring_norm) * ENCODER_CORRECTION.ankle_spring_err_std
        + ENCODER_CORRECTION.ankle_spring_err_mean
    )
    ankle_spring_angle = spring_lin + spring_err

    # ankle torque
    ankle_torque = ANKLE.stiffness * (ankle_motor_angle - ankle_spring_angle)

    return SensorReadings(
        knee_motor_angle=knee_motor_angle,
        knee_angle=knee_angle,
        knee_torque=knee_torque,
        ankle_motor_angle=ankle_motor_angle,
        ankle_spring_angle=ankle_spring_angle,
        ankle_angle=ankle_angle,
        ankle_torque=ankle_torque,
    )

def main():
    ec = EtherCATInterface("eth0")
    ec.connect()
    print(f"Recording for {DURATION} seconds")

    t_log = []
    knee = {"motor angle": [], "joint angle": []}
    ankle = {"motor angle": [], "spring angle": [], "joint angle": []}

    period = 0.001   # target loop period (s) -> ~1 kHz
    t0 = time.perf_counter()
    next_t = t0
    try:
        while (t := time.perf_counter() - t0) < DURATION:
            next_t += period
            r = _convert_to_readings(ec.read())
            t_log.append(t)
            knee["motor angle"].append(degrees(r.knee_motor_angle))
            knee["joint angle"].append(degrees(r.knee_angle))
            ankle["motor angle"].append(degrees(r.ankle_motor_angle))
            ankle["spring angle"].append(degrees(r.ankle_spring_angle))
            ankle["joint angle"].append(degrees(r.ankle_angle))
            # sleep only the time left in this period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("terminated early")
    finally:
        ec.close()

    out = make_run_dir("read_encoders")
    _plot("Knee", t_log, knee, out)
    _plot("Ankle", t_log, ankle, out)

if __name__ == "__main__":
    main()

    