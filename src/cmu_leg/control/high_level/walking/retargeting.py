"""Knee angle remap between our leg and the symmetric 'human' leg.

Two controllers are built on human data and need the same map: the
neuromuscular stance model, whose muscle tables assume human segment lengths,
and the NN swing predictor, which was trained on able-bodied gait. Our leg is
0.66 m thigh / 0.215 m shank, so both run on a virtual leg with both segments
equal to (thigh + shank)/2.

The map holds the chord |hip -> ankle| equal between the two legs and solves
the law of cosines for the virtual knee. Knee 0 is straight and flexion is
positive on both legs, so there is no offset or unit change here.

NeuromuscularStance needs the forward map only, and wraps it with its own ankle
remap (it additionally preserves how the foot sits relative to the leg axis).
The NN predictor needs both directions.
"""
import math

import numpy as np

from cmu_leg.config.params import LEG

L_THIGH = LEG.hip_to_knee               # 0.66 m
L_SHANK = LEG.shank_length              # 0.215 m
L_HUMAN = 0.5 * (L_THIGH + L_SHANK)     # symmetric virtual leg, 0.4375 m each


def _clamp_unit(x):
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


def chord(knee_angle: float) -> float:
    """Distance from hip to ankle on our leg.

    Args:
        knee_angle: scalar [rad], 0 straight, flexion positive.

    Returns:
        scalar [m].

    Knee 0 is straight, so the triangle's interior angle at the knee is
    pi - knee_angle and cos(pi - knee) = -cos(knee), hence the + sign.
    """
    return math.sqrt(L_THIGH**2 + L_SHANK**2
                     + 2.0 * L_THIGH * L_SHANK * math.cos(knee_angle))


def knee_leg_to_human(knee_angle: float) -> float:
    """Map our knee angle onto the virtual leg.

    Args:
        knee_angle: scalar [rad] in our leg's frame.

    Returns:
        scalar [rad] on the virtual leg. Signed, so a hyperextended (negative)
        knee maps to a negative virtual knee.
    """
    if knee_angle == 0.0:
        return 0.0
    l3 = chord(knee_angle)
    ck = (2.0 * L_HUMAN**2 - l3 * l3) / (2.0 * L_HUMAN**2)
    return math.copysign(math.pi - math.acos(_clamp_unit(ck)), knee_angle)


def knee_human_to_leg(knee_human, max_soft: float):
    """Map the virtual leg's knee angle back onto ours.

    Args:
        knee_human: scalar or array [rad] on the virtual leg. Arrays pass
            through elementwise, which is how the network's 41-point
            prediction is converted in one call.
        max_soft: upper clamp [rad] applied to the result.

    Returns:
        Same shape as knee_human, [rad] in our leg's frame, in [0, max_soft].

    Two clamps, both from the baseline. cos of the leg knee is capped at 0,
    which caps the result at 90 degrees and keeps the arccos in range for
    virtual-leg angles our shorter shank cannot reach; max_soft then applies
    the joint's own limit.
    """
    l3sq = (2.0 * L_HUMAN**2
            - 2.0 * L_HUMAN**2 * np.cos(np.pi - np.asarray(knee_human)))
    ck = (L_THIGH**2 + L_SHANK**2 - l3sq) / (2.0 * L_THIGH * L_SHANK)
    ck = np.clip(ck, -1.0, 0.0)
    return np.minimum(np.pi - np.arccos(ck), max_soft)
