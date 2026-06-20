"""read_encoders: raw EtherCAT counts -> engineering-unit sensor readings.

Pure, stateless port of the Simulink ``read_encoders`` model. Every formula
below was reconstructed block-by-block from that model and validated against a
real recording (tests/knee.mat, tests/ankle.mat): the SEA torque formulas and
the ankle-motor angle reproduce the recording to machine precision. The joint
angle constants live in ``EncoderCalibration``; the *offsets* are per-session
encoder zeros (from spring_zero_positions.mat by default) and must match the
session that produced any recording you verify against.

Block map (Simulink read_encoders):
  knee motor angle  = AbsCounts2Rad( Bias1(knee motor counts) )
  knee angle        = "Convert to radians"  (linear + normalized poly err-correct)
  knee torque       = knee_stiffness * (knee motor angle - knee angle)
  ankle motor angle = AbsCounts2Rad1( Bias3(ankle motor counts) )
  ankle angle       = ankle_abs_counts_to_rad * counts + ankle_angle_offset
  ankle spring angle= "Convert to radians1" (counts->rad poly + normalized poly err-correct)
  ankle torque      = ankle_stiffness * (ankle motor angle - ankle spring angle)
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, pi

from .ethercat_io import RawSensorFrame

TWO_PI = 2.0 * pi


# ---------------------------------------------------------------------------
# Calibration. Values from baseline_controller params.m + the .mat files
# (spring_zero_positions, encoder_counts_to_rad_polys, ankle_angle_offset).
# Polynomials are highest-power-first (MATLAB/numpy polyval convention).
# NOTE: the *_offset fields are per-session encoder zeros. The defaults are
# from spring_zero_positions.mat; a given recording may have used different
# zeros (the motors are re-zeroed each session) — inject a matching
# EncoderCalibration to verify angles bit-exactly.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EncoderCalibration:
    # Motor abs encoders (linear, 2^30 counts/rev)
    motor_counts_to_rad: float = TWO_PI / 2 ** 30
    knee_motor_offset: int = -2_635          # kneemotor0pos
    ankle_motor_offset: int = -951           # anklemotor0pos

    # Knee joint: linear term + polynomial error correction on normalized counts
    knee_counts_to_rad: float = -1.481916095218253e-08
    knee_joint_offset: int = 234_149_441     # knee0pos
    knee_counts_mean: float = -49_248_866.40327176
    knee_counts_std: float = 33_291_983.071372624
    knee_err_poly: tuple[float, ...] = (0.47774077, -0.38791886, -0.80362217, 0.50741695)
    knee_err_mean: float = -6.0769758202966855e-06
    knee_err_std: float = 0.00037360483505769084

    # Ankle joint angle: linear scale + fixed angle offset
    ankle_counts_to_rad: float = TWO_PI / 2 ** 17
    ankle_joint_offset: int = -553           # ankle0pos
    ankle_angle_offset: float = 0.0444       # ankle_angle_offset.mat; NOT applied
                                             # in read_encoders (downstream only)
    ankle_total_counts: int = 2 ** 17        # wrap modulus

    # Ankle spring angle: counts->rad poly + polynomial error correction
    ankle_spring_counts_to_rad_poly: tuple[float, ...] = (
        -1.27664165e-15, 2.81074451e-10, 4.75918169e-05, 0.0,
    )
    ankle_spring_counts_mean: float = -1248.7238249778297
    ankle_spring_counts_std: float = 7852.009105751588
    ankle_spring_err_poly: tuple[float, ...] = (
        2.37075304, -0.26480173, -4.05764862, -0.08075658, 0.56996956,
    )
    ankle_spring_err_mean: float = -0.00038694562088488905
    ankle_spring_err_std: float = 0.0008505143990917394

    # SEA stiffness (Nm/rad)
    knee_stiffness: float = 1666.0
    ankle_stiffness: float = 993.8613


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


CAL = EncoderCalibration()


def _polyval(coeffs: tuple[float, ...], x: float) -> float:
    """Horner evaluation, highest-power-first (matches MATLAB/numpy polyval)."""
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


def _wrap_counts(counts: int, total: int) -> int:
    """Wrap absolute counts into [-total/2, total/2). No-op when already in range
    (so feeding pre-wrapped recorded counts is safe). Simulink "Wrap to ..." block.
    """
    return counts - total * floor((counts + total / 2) / total)


def read_encoders(frame: RawSensorFrame, cal: EncoderCalibration = CAL) -> SensorReadings:
    """Convert one raw PDO frame into engineering-unit sensor readings."""
    # --- knee motor angle (linear abs encoder) ---
    knee_motor_angle = cal.motor_counts_to_rad * (frame.knee_motor_counts - cal.knee_motor_offset)

    # --- knee joint angle: linear + normalized polynomial error correction ---
    kc = frame.knee_joint_counts - cal.knee_joint_offset
    knee_lin = cal.knee_counts_to_rad * kc
    knee_norm = (kc - cal.knee_counts_mean) / cal.knee_counts_std
    knee_err = _polyval(cal.knee_err_poly, knee_norm) * cal.knee_err_std + cal.knee_err_mean
    knee_angle = knee_lin + knee_err

    # --- knee torque (SEA: motor minus joint) ---
    knee_torque = cal.knee_stiffness * (knee_motor_angle - knee_angle)

    # --- ankle motor angle (linear abs encoder) ---
    ankle_motor_angle = cal.motor_counts_to_rad * (frame.ankle_motor_counts - cal.ankle_motor_offset)

    # --- ankle joint counts: wrap, then offset ---
    awc = _wrap_counts(frame.ankle_joint_counts, cal.ankle_total_counts)
    ac = awc - cal.ankle_joint_offset

    # ankle joint angle (linear scale). The Simulink read_encoders .slx adds
    # ankle_angle_offset here, but the hardware recording shows it is NOT applied
    # in read_encoders — it's a Get-Prosthesis-State alignment offset used
    # downstream. Verified bit-exact without it (tests/test_encoders_offline.py).
    ankle_angle = cal.ankle_counts_to_rad * ac

    # ankle spring angle: counts->rad poly + normalized polynomial error correction
    spring_lin = _polyval(cal.ankle_spring_counts_to_rad_poly, ac)
    spring_norm = (ac - cal.ankle_spring_counts_mean) / cal.ankle_spring_counts_std
    spring_err = (
        _polyval(cal.ankle_spring_err_poly, spring_norm) * cal.ankle_spring_err_std
        + cal.ankle_spring_err_mean
    )
    ankle_spring_angle = spring_lin + spring_err

    # --- ankle torque (SEA: motor minus spring) ---
    ankle_torque = cal.ankle_stiffness * (ankle_motor_angle - ankle_spring_angle)

    return SensorReadings(
        knee_motor_angle=knee_motor_angle,
        knee_angle=knee_angle,
        knee_torque=knee_torque,
        ankle_motor_angle=ankle_motor_angle,
        ankle_spring_angle=ankle_spring_angle,
        ankle_angle=ankle_angle,
        ankle_torque=ankle_torque,
    )
