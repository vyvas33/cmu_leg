# Ankle velocity saturation. Final safety stage before the Elmo drive.
# Reads from prosthesis state and the controller's desired velocity + raw torque
# feed-forward. Produces:
#   - ankle_velocity_commanded   (sent to Elmo as velocity setpoint)
#   - ankle_motor_torque_ff      (sent to Elmo as torque feed-forward)
#
# Simulink source block: "Ankle Desired Saturation" (renamed here to "Ankle
# Velocity Saturation" to mirror the knee pipeline).
#
# Behavior (from baseline_impedance_min_jerk, Ankle Desired Saturation block):
#
#   flexion_violation   = (ankle_motor_angle > ANKLE_ANGLE_MAX)     # hard limit
#   extension_violation = (ankle_motor_angle < ANKLE_ANGLE_MIN)     # hard limit
#
#   velocity_upper = 0 if flexion_violation   else +ANKLE_VEL_MAX
#   velocity_lower = 0 if extension_violation else -ANKLE_VEL_MAX
#   torque_upper   = 0 if flexion_violation   else +ANKLE_TORQUE_MAX
#   torque_lower   = 0 if extension_violation else -ANKLE_TORQUE_MAX
#
#   velocity_sat  = clip(ankle_velocity_desired, velocity_lower, velocity_upper)
#   torque_ff_sat = clip(ankle_motor_torque_ff,  torque_lower,   torque_upper)
#
# The saturated signals then pass through a 50 Hz / order-2 Butterworth lowpass
# (separate state per channel) before reaching the drive.
#
# Differences vs. KneeVelocitySaturation:
#   - LPF cutoff is 50 Hz (knee uses 75 Hz).
#   - BOTH violation checks use ankle_motor_angle (knee used motor angle for
#     extension but joint angle for flexion).

from cmu_leg.types.state import ProsthesisState
from cmu_leg.control.utils import Lowpass
import cmu_leg.config.params as params
import numpy as np


class AnkleVelocitySaturation:
    def __init__(self):
        self.ankle_vel_max = params.ANKLE_VEL_MAX
        self.ankle_torque_max = params.ANKLE_TORQUE_MAX
        self.ankle_angle_min = params.ANKLE_ANGLE_MIN
        self.ankle_angle_max = params.ANKLE_ANGLE_MAX

        self._lowpass_velocity = Lowpass(
            cutoff_freq=params.ANKLE_VEL_SAT_LOWPASS_CUTOFF,
            sample_rate=params.SAMPLE_RATE,
            order=params.ANKLE_VEL_SAT_LOWPASS_ORDER,
        )
        self._lowpass_torque_ff = Lowpass(
            cutoff_freq=params.ANKLE_VEL_SAT_LOWPASS_CUTOFF,
            sample_rate=params.SAMPLE_RATE,
            order=params.ANKLE_VEL_SAT_LOWPASS_ORDER,
        )

        # change to True once everything is verified as safe to use.
        self._verified = False

    # Reset the two lowpass filters so the next compute() does not emit a
    # startup transient. Called by Low Level Control on the disabled -> enabled
    # edge, typically with the current desired values (or 0).
    def reset(self,
              ankle_velocity_desired_init: float = 0.0,
              ankle_motor_torque_ff_desired_init: float = 0.0) -> None:
        self._lowpass_velocity.reset(ankle_velocity_desired_init)
        self._lowpass_torque_ff.reset(ankle_motor_torque_ff_desired_init)

    def compute(self,
                state: ProsthesisState,
                ankle_velocity_desired: float,
                ankle_motor_torque_ff_desired: float) -> tuple[float, float]:

        if not self._verified:
            raise NotImplementedError("Ankle Velocity Saturation not verified yet!")

        flexion_violation = state.ankle_motor_angle > self.ankle_angle_max
        extension_violation = state.ankle_motor_angle < self.ankle_angle_min

        velocity_upper = 0.0 if flexion_violation else self.ankle_vel_max
        velocity_lower = 0.0 if extension_violation else -self.ankle_vel_max
        torque_upper = 0.0 if flexion_violation else self.ankle_torque_max
        torque_lower = 0.0 if extension_violation else -self.ankle_torque_max

        velocity_saturated = np.clip(ankle_velocity_desired,
                                     velocity_lower, velocity_upper)
        torque_ff_saturated = np.clip(ankle_motor_torque_ff_desired,
                                      torque_lower, torque_upper)

        ankle_velocity_commanded = self._lowpass_velocity.compute(velocity_saturated)
        ankle_motor_torque_ff = self._lowpass_torque_ff.compute(torque_ff_saturated)

        return ankle_velocity_commanded, ankle_motor_torque_ff
