"""Sagittal-plane forward kinematics, hip -> knee -> ankle -> toe/heel.

Usage:
    from cmu_leg.estimation.leg_kinematics import leg_points

    pts = leg_points(state.hip_angle, state.knee_angle, state.ankle_angle)
    pts.knee, pts.ankle, pts.toe, pts.heel     # (x, z) metres, +x forward, +z up

Angles are radians in the ProsthesisState convention: hip_angle from vertical,
knee/ankle from the encoders, all flexion-positive. The hip is the root and
defaults to the origin -- three joint angles fix the leg's shape, not its
position in the world.
"""

from typing import NamedTuple

import numpy as np

from cmu_leg.config.params import LEG


class LegPoints(NamedTuple):
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    toe: np.ndarray
    heel: np.ndarray


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


def leg_points(hip_angle: float, knee_angle: float, ankle_angle: float,
               hip=(0.0, 0.0)) -> LegPoints:
    hip = np.asarray(hip, dtype=float)

    R_th = _rot(-hip_angle)
    R_sh = _rot(knee_angle - hip_angle)
    R_fo = _rot(knee_angle + ankle_angle - hip_angle)

    knee = hip + R_th @ np.array([0.0, -LEG.hip_to_knee])
    ankle = knee + R_sh @ np.array([0.0, -LEG.shank_length])
    toe = ankle + R_fo @ np.array([LEG.ankle_to_toe[0], LEG.ankle_to_toe[2]])
    heel = ankle + R_fo @ np.array([LEG.ankle_to_heel[0], LEG.ankle_to_heel[2]])

    return LegPoints(hip, knee, ankle, toe, heel)


def eskf_chain_residual(hip_angle: float, knee_angle: float,
                        ankle_angle: float) -> float:
    """Largest disagreement [m] between this chain and the ESKF's.

    This runs hip-down, the ESKF's fk_p_fo_th runs foot-up. Both must place the
    thigh IMU at the same point relative to the ankle. Should be ~1e-16, and is
    independent of hip_angle.
    """
    from cmu_leg.estimation.eskf.measurements import (
        fk_p_fo_th, P_SH_TH, S_ANKLE, S_KNEE, THETA0_ANKLE, THETA0_KNEE)

    pts = leg_points(hip_angle, knee_angle, ankle_angle)
    R_th = _rot(-hip_angle)
    R_fo = _rot(knee_angle + ankle_angle - hip_angle)

    thigh_imu = pts.knee + R_th @ np.array([P_SH_TH[0], P_SH_TH[2]])
    mine = R_fo.T @ (thigh_imu - pts.ankle)

    a = S_ANKLE * ankle_angle + THETA0_ANKLE
    k = S_KNEE * knee_angle + THETA0_KNEE
    theirs = fk_p_fo_th(a, k)[[0, 2]]

    return float(np.abs(mine - theirs).max())


HEEL_RADIUS = float(np.hypot(LEG.ankle_to_heel[0], LEG.ankle_to_heel[2]))
HEEL_ANGLE = float(np.arctan2(-LEG.ankle_to_heel[0], -LEG.ankle_to_heel[2]))


def heel_forward(hip_angle: float, knee_angle: float, hip_x: float,
                 hip_to_knee: float = LEG.hip_to_knee,
                 heel_radius: float = HEEL_RADIUS,
                 heel_angle: float = HEEL_ANGLE) -> float:
    """Horizontal position of the heel [m], in the hip's frame.

    Args:
        hip_angle, knee_angle: scalars [rad], ProsthesisState convention.
        hip_x: hip's horizontal position [m], which the result is relative to.
        hip_to_knee: thigh length [m]. Subject specific, so it is exposed
            rather than always taken from LEG.
        heel_radius, heel_angle: the heel's polar offset from the shank, [m]
            and [rad]. Defaulted from LEG.ankle_to_heel, but exposed because
            the baseline carries its own measured radius that does not equal
            its own components' hypotenuse.

    Returns:
        scalar [m], +x forward.

    The ankle angle is deliberately absent: this places the heel relative to the
    shank, treating the foot as rigid with it. That is what the baseline's
    `get_heel_forward_single` does, and the swing phase chart compares the
    result against hip_x, so the two have to agree. `leg_points` is the full
    chain and does use the ankle.
    """
    shank = knee_angle - hip_angle
    return (hip_x
            + hip_to_knee * np.sin(hip_angle)
            - LEG.shank_length * np.sin(shank)
            - heel_radius * np.sin(shank + heel_angle))
