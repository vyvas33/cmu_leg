"""
RawEtherCATFrame to EncoderReadings. Stateless script to get usable encoder data.

Usage: 
    from cmu_leg.estimation.encoders import convert_encoders
    encoder_readings = convert_encoders(raw_encoder_ints)

"""

from math import floor

from cmu_leg.types.sensors import RawEtherCATFrame, EncoderReadings
from cmu_leg.config.params import ENCODERS, ENCODER_CORRECTION, KNEE, ANKLE


def _polyval(coeffs: tuple[float, ...], x: float) -> float:
    """Horner evaluation, highest-power-first (matches MATLAB/numpy polyval)."""
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc

def _wrap_counts(counts: int, total: int) -> int:
    """Wrap absolute counts into [-total/2, total/2)."""
    return counts - total * floor((counts + total / 2) / total)

def convert_encoders(frame: RawEtherCATFrame) -> EncoderReadings:
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

    return EncoderReadings(
        knee_motor_angle=knee_motor_angle,
        knee_angle=knee_angle,
        knee_torque=knee_torque,
        ankle_motor_angle=ankle_motor_angle,
        ankle_spring_angle=ankle_spring_angle,
        ankle_angle=ankle_angle,
        ankle_torque=ankle_torque,
    )
