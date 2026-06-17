from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired
from cmu_leg.control.utils import Lowpass
import cmu_leg.config.params as params
from cmu_leg.config.params import KneeParams, CommonParams
import numpy as np


# This class takes in the state and the desired torque from high level controller, computes the commanded torque and sends it to knee torque controller.
class KneeTorqueConditioning:
    def __init__(self, knee: KneeParams = params.KNEE, common: CommonParams = params.COMMON):
        self.knee_torque_max = knee.torque_max
        self.knee_angle_max_soft = knee.angle_max_soft
        self.knee_angle_min_soft = knee.angle_min_soft
        self.c_jointstop = knee.c_jointstop
        self.w_max_jointstop = knee.w_max_jointstop
        self.damping_gain = common.damping_gain
        self.knee_up_limit = knee.up_limit

        # `ext` is UNRESOLVED. AnkleTorqueConditioning already uses per-stop sign
        # multipliers (knee.flex_ext = +1 at the upper stop, knee.extend_ext = -1
        # at the lower). This knee block still applies a single shared `ext` in
        # BOTH _flexion_stop and _extension_stop — the bug flagged at the bottom
        # of this file. Splitting it is a behavioral change pending Simulink
        # verification, so `ext` is deliberately left None: __init__ succeeds, but
        # compute() is unreachable behind the _verified gate until this is
        # reconciled. See [[low-level-conditioning-ext-pending]].
        self.ext = None  # TODO(ext): resolve flex vs extend signs (knee.flex_ext / knee.extend_ext)
        self._lowpass = Lowpass()

        # change this to true once everything is verified as safe to use.
        self._verified = False

    def reset(self) -> None:
        self._lowpass.reset(0.0)

    def compute(self, state: ProsthesisState, desired_torque: TorqueDesired):

        if not self._verified:
            raise NotImplementedError("Knee Torque Conditioning not verified yet!")
        
        flexion_stop_torque = self._flexion_stop(state.knee_angle, state.knee_velocity)
        extension_stop_torque = self._extension_stop(state.knee_motor_angle, state.knee_motor_velocity)
    
        if state.knee_angle > self.knee_angle_max_soft:
            up = flexion_stop_torque
        else: up = self.knee_torque_max

        if state.knee_motor_angle < self.knee_angle_min_soft and not state.standing:
            lo = extension_stop_torque
        else: lo = - self.knee_torque_max

        torque_input = desired_torque.knee_desired + flexion_stop_torque + extension_stop_torque

        saturated_torque = self._torque_saturation(torque_input, lo, up)

        damping_compensation = self.damping_gain * state.knee_velocity
        commanded_torque = self._lowpass.compute(saturated_torque) + damping_compensation

        return commanded_torque

    def _flexion_stop(self, knee_position, knee_velocity): 
        e_knee_position = self.ext * self.c_jointstop * (knee_position - self.knee_up_limit)

        e_knee_velocity = - self.ext * (1/self.w_max_jointstop) * knee_velocity 

        if e_knee_position > 0 and e_knee_velocity < 1:
            stop_torque = (-1) * e_knee_position * (1 - e_knee_velocity)
        else:
            stop_torque = 0.0

        return self.ext * stop_torque

    def _extension_stop(self, knee_motor_position, knee_motor_velocity):
        e_knee_position = self.ext * self.c_jointstop * (knee_motor_position - self.knee_up_limit)

        e_knee_velocity = - self.ext * (1/self.w_max_jointstop) * knee_motor_velocity 

        if e_knee_position > 0 and e_knee_velocity < 1:
            stop_torque = (-1) * e_knee_position * (1 - e_knee_velocity)
        else:
            stop_torque = 0.0

        return self.ext * stop_torque
        
    def _torque_saturation(self, torque_input, lo, up):

        return np.clip(torque_input, lo, up)

        
# things to verify - Extension and Flexion up limit angles and signs (right now the blocks are identical with same parameters in simulink - bug?)
