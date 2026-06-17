# Knee velocity saturation. Final safety stage before the Elmo drive.
# Reads from prosthesis state and the controller's desired velocity + raw torque
# feed-forward. Produces:
#   - knee_velocity_commanded   (sent to Elmo as velocity setpoint)
#   - knee_motor_torque_ff      (sent to Elmo as torque feed-forward)
#
# Behavior (from baseline_impedance_min_jerk, Knee Velocity Saturation block):
#
#   flexion_violation   = (knee_angle       > KNEE_ANGLE_MAX)     # hard limit
#   extension_violation = (knee_motor_angle < KNEE_ANGLE_MIN)     # hard limit
#
#   velocity_upper = 0 if flexion_violation   else +KNEE_VEL_MAX
#   velocity_lower = 0 if extension_violation else -KNEE_VEL_MAX
#   torque_upper   = 0 if flexion_violation   else +KNEE_TORQUE_MAX
#   torque_lower   = 0 if extension_violation else -KNEE_TORQUE_MAX
#
#   velocity_sat  = clip(knee_velocity_desired, velocity_lower, velocity_upper)
#   torque_ff_sat = clip(knee_motor_torque_ff,  torque_lower,   torque_upper)
#
# The saturated signals then pass through a 75 Hz / order-2 Butterworth lowpass
# (separate state per channel) to smooth the clipping discontinuity before it
# reaches the drive.
#
# Notes vs Conditioning:
#   - This block uses HARD limits (knee_angle_min / knee_angle_max).
#     Conditioning uses SOFT limits and applies a graded backstop torque.
#     This block is the last-line zero-the-command safety net.
#   - Flexion is gated on the JOINT angle; extension is gated on the MOTOR
#     angle. Same asymmetry as in Conditioning.

from cmu_leg.types.state import ProsthesisState
from cmu_leg.control.utils import Lowpass
import cmu_leg.config.params as params
from cmu_leg.config.params import KneeParams, CommonParams
import numpy as np


class KneeVelocitySaturation:
    def __init__(self, knee: KneeParams = params.KNEE, common: CommonParams = params.COMMON):
        self.knee_vel_max = knee.vel_max
        self.knee_torque_max = knee.torque_max
        self.knee_angle_min = knee.angle_min
        self.knee_angle_max = knee.angle_max

        self._lowpass_velocity = Lowpass(
            cutoff_freq=knee.vel_sat_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=knee.vel_sat_lowpass_order,
        )
        self._lowpass_torque_ff = Lowpass(
            cutoff_freq=knee.vel_sat_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=knee.vel_sat_lowpass_order,
        )

        # change to True once everything is verified as safe to use.
        self._verified = False

    # Reset the two lowpass filters so the next compute() does not emit a
    # startup transient. Called by Low Level Control on the disabled -> enabled
    # edge, typically with the current desired values (or 0).
    def reset(self,
              knee_velocity_desired_init: float = 0.0,
              knee_motor_torque_ff_desired_init: float = 0.0) -> None:
        self._lowpass_velocity.reset(knee_velocity_desired_init)
        self._lowpass_torque_ff.reset(knee_motor_torque_ff_desired_init)

    def compute(self,
                state: ProsthesisState,
                knee_velocity_desired: float,
                knee_motor_torque_ff_desired: float) -> tuple[float, float]:

        if not self._verified:
            raise NotImplementedError("Knee Velocity Saturation not verified yet!")

        flexion_violation = state.knee_angle > self.knee_angle_max
        extension_violation = state.knee_motor_angle < self.knee_angle_min

        velocity_upper = 0.0 if flexion_violation else self.knee_vel_max
        velocity_lower = 0.0 if extension_violation else -self.knee_vel_max
        torque_upper = 0.0 if flexion_violation else self.knee_torque_max
        torque_lower = 0.0 if extension_violation else -self.knee_torque_max

        velocity_saturated = np.clip(knee_velocity_desired,
                                     velocity_lower, velocity_upper)
        torque_ff_saturated = np.clip(knee_motor_torque_ff_desired,
                                      torque_lower, torque_upper)

        knee_velocity_commanded = self._lowpass_velocity.compute(velocity_saturated)
        knee_motor_torque_ff = self._lowpass_torque_ff.compute(torque_ff_saturated)

        return knee_velocity_commanded, knee_motor_torque_ff
