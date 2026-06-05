from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired
from cmu_leg.control.utils import Lowpass
import cmu_leg.config.params as params
import numpy as np


# This class takes in the state and the desired torque from high level controller, computes the commanded torque and sends it to ankle torque controller.
# Simulink source block: "Ankle Torque Saturation" (renamed here to "Ankle
# Torque Conditioning" to mirror the knee pipeline).
#
# Differences vs. KneeTorqueConditioning:
#   - No additive stop torques: desired torque is clipped, not summed with stops.
#   - No FF damping compensation.
#   - Both stops use motor-side (motor_angle, LPF(motor_velocity)).
#   - Per-direction `ext` sign and `phi_up_limit` (Simulink mask values).
class AnkleTorqueConditioning:
    def __init__(self):
        self.ankle_torque_max = params.ANKLE_TORQUE_MAX
        self.ankle_angle_max_soft = params.ANKLE_ANGLE_MAX_SOFT
        self.ankle_angle_min_soft = params.ANKLE_ANGLE_MIN_SOFT
        self.c_jointstop = params.ANKLE_C_JOINTSTOP
        self.w_max_jointstop = params.ANKLE_W_MAX_JOINTSTOP
        self.flex_ext = params.ANKLE_FLEX_EXT       # +1, used in flexion stop
        self.extend_ext = params.ANKLE_EXTEND_EXT   # -1, used in extension stop

        self._lowpass_motor_velocity = Lowpass()    # 25 Hz default
        self._lowpass_out = Lowpass()               # 25 Hz default

        # change this to true once everything is verified as safe to use.
        self._verified = False

    def reset(self) -> None:
        self._lowpass_motor_velocity.reset(0.0)
        self._lowpass_out.reset(0.0)

    def compute(self, state: ProsthesisState, desired_torque: TorqueDesired):

        if not self._verified:
            raise NotImplementedError("Ankle Torque Conditioning not verified yet!")

        # filter motor velocity once; both stops consume the filtered value
        ankle_motor_velocity_filtered = self._lowpass_motor_velocity.compute(state.ankle_motor_velocity)

        flexion_stop_torque = self._flexion_stop(state.ankle_motor_angle, ankle_motor_velocity_filtered)
        extension_stop_torque = self._extension_stop(state.ankle_motor_angle, ankle_motor_velocity_filtered)

        if state.ankle_motor_angle > self.ankle_angle_max_soft:
            up = flexion_stop_torque
        else: up = self.ankle_torque_max

        if state.ankle_motor_angle < self.ankle_angle_min_soft:
            lo = extension_stop_torque
        else: lo = - self.ankle_torque_max

        saturated_torque = self._torque_saturation(desired_torque.ankle_desired, lo, up)
        commanded_torque = self._lowpass_out.compute(saturated_torque)

        return commanded_torque

    def _flexion_stop(self, ankle_motor_position, ankle_motor_velocity):
        e_ankle_position = self.flex_ext * self.c_jointstop * (ankle_motor_position - self.ankle_angle_max_soft)

        e_ankle_velocity = - self.flex_ext * (1/self.w_max_jointstop) * ankle_motor_velocity

        if e_ankle_position > 0 and e_ankle_velocity < 1:
            stop_torque = (-1) * e_ankle_position * (1 - e_ankle_velocity)
        else:
            stop_torque = 0.0

        return self.flex_ext * stop_torque

    def _extension_stop(self, ankle_motor_position, ankle_motor_velocity):
        e_ankle_position = self.extend_ext * self.c_jointstop * (ankle_motor_position - self.ankle_angle_min_soft)

        e_ankle_velocity = - self.extend_ext * (1/self.w_max_jointstop) * ankle_motor_velocity

        if e_ankle_position > 0 and e_ankle_velocity < 1:
            stop_torque = (-1) * e_ankle_position * (1 - e_ankle_velocity)
        else:
            stop_torque = 0.0

        return self.extend_ext * stop_torque

    def _torque_saturation(self, torque_input, lo, up):

        return np.clip(torque_input, lo, up)
