"""Ankle low-level pipeline: conditioning -> torque controller -> velocity
saturation. Three independent stages, each with its own `_verified` gate and
reset, matching the nm_nn_swing Ankle subsystems. Kept as separate classes in
one file so the per-stage verification/reset boundaries stay intact.
"""

import cmu_leg.config.params as params
from cmu_leg.config.params import AnkleParams, CommonParams
from cmu_leg.control.utils import Lowpass, BackwardDifference, clamp
from cmu_leg.types.commands import TorqueDesired
from cmu_leg.types.state import ProsthesisState


# This class takes in the state and the desired torque from high level controller, computes the commanded torque and sends it to ankle torque controller.
# Simulink source block: "Ankle Torque Saturation" (renamed here to "Ankle
# Torque Conditioning" to mirror the knee pipeline).
class AnkleTorqueConditioning:
    def __init__(self, ankle: AnkleParams = params.ANKLE):
        self.ankle_torque_max = ankle.torque_max
        self.ankle_angle_max_soft = ankle.angle_max_soft
        self.ankle_angle_min_soft = ankle.angle_min_soft
        self.c_jointstop = ankle.c_jointstop
        self.w_max_jointstop = ankle.w_max_jointstop
        self.flex_ext = ankle.flex_ext  # +1, used in flexion stop
        self.extend_ext = ankle.extend_ext  # -1, used in extension stop

        self._lowpass_motor_velocity = Lowpass()  # 25 Hz default
        self._lowpass_out = Lowpass()  # 25 Hz default

        # change this to true once everything is verified as safe to use.
        self._verified = True

    def reset(self) -> None:
        self._lowpass_motor_velocity.reset(0.0)
        self._lowpass_out.reset(0.0)

    def compute(self, state: ProsthesisState, desired_torque: TorqueDesired):

        if not self._verified:
            raise NotImplementedError("Ankle Torque Conditioning not verified yet!")

        # filter motor velocity once; both stops consume the filtered value
        ankle_motor_velocity_filtered = self._lowpass_motor_velocity.compute(
            state.ankle_motor_velocity
        )

        flexion_stop_torque = self._flexion_stop(
            state.ankle_motor_angle, ankle_motor_velocity_filtered
        )
        extension_stop_torque = self._extension_stop(
            state.ankle_motor_angle, ankle_motor_velocity_filtered
        )

        if state.ankle_motor_angle > self.ankle_angle_max_soft:
            up = flexion_stop_torque
        else:
            up = self.ankle_torque_max

        if state.ankle_motor_angle < self.ankle_angle_min_soft:
            lo = extension_stop_torque
        else:
            lo = -self.ankle_torque_max

        saturated_torque = self._torque_saturation(desired_torque.ankle_desired, lo, up)
        commanded_torque = self._lowpass_out.compute(saturated_torque)

        return commanded_torque

    def _flexion_stop(self, ankle_motor_position, ankle_motor_velocity):
        e_ankle_position = (
            self.flex_ext
            * self.c_jointstop
            * (ankle_motor_position - self.ankle_angle_max_soft)
        )

        e_ankle_velocity = (
            -self.flex_ext * (1 / self.w_max_jointstop) * ankle_motor_velocity
        )

        if e_ankle_position > 0 and e_ankle_velocity < 1:
            stop_torque = (-1) * e_ankle_position * (1 - e_ankle_velocity)
        else:
            stop_torque = 0.0

        return self.flex_ext * stop_torque

    def _extension_stop(self, ankle_motor_position, ankle_motor_velocity):
        e_ankle_position = (
            self.extend_ext
            * self.c_jointstop
            * (ankle_motor_position - self.ankle_angle_min_soft)
        )

        e_ankle_velocity = (
            -self.extend_ext * (1 / self.w_max_jointstop) * ankle_motor_velocity
        )

        if e_ankle_position > 0 and e_ankle_velocity < 1:
            stop_torque = (-1) * e_ankle_position * (1 - e_ankle_velocity)
        else:
            stop_torque = 0.0

        return self.extend_ext * stop_torque

    def _torque_saturation(self, torque_input, lo, up):

        return clamp(torque_input, lo, up)


# Ankle torque controller. Reads from prosthesis state and the commanded torque
# coming out of conditioning. Produces:
#   - desired motor velocity
#   - torque feed-forward
#
# Equations (from baseline_impedance_min_jerk, Ankle Torque Controller block):
#   tau_dot               = d/dt  tau_cmd                # backward difference
#   tau_ddot              = d/dt  tau_dot                # backward difference
#   spring_velocity_filt  = LPF_20(state.ankle_spring_velocity)
#
#   omega_des = spring_velocity_filt + tau_dot / Ks + Kp * (tau_cmd - tau_meas)
#   tau_ff    = tau_cmd + (bm/Ks)*tau_dot + (Jm/Ks)*tau_ddot
#
class AnkleTorqueController:
    def __init__(self, ankle: AnkleParams = params.ANKLE, common: CommonParams = params.COMMON):
        self.ankle_kp = ankle.kp
        self.ankle_stiffness = ankle.stiffness
        self.ankle_motor_damping = ankle.motor_damping
        self.ankle_motor_inertia = ankle.motor_inertia

        # Both differences run on the caller's measured dt. These two are the
        # most dt-sensitive quantities in the stack: tau_ddot is a difference OF
        # tau_dot, so a bad dt is compounded through both stages rather than
        # just added, and the result feeds the drive feedforward.
        self._dd_tau = BackwardDifference()
        self._dd_tau_dot = BackwardDifference()
        self._lowpass_spring_velocity = Lowpass(
            cutoff_freq=ankle.torque_ctrl_spring_vel_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=ankle.torque_ctrl_spring_vel_lowpass_order,
        )

        # change to True once everything is verified as safe to use.
        self._verified = True

    # Reset derivative + lowpass state so the next compute() does not emit a step.
    def reset(self,
              ankle_torque_commanded_init: float = 0.0,
              ankle_spring_velocity_init: float = 0.0) -> None:
        self._dd_tau.reset(ankle_torque_commanded_init)
        self._dd_tau_dot.reset(0.0)
        self._lowpass_spring_velocity.reset(ankle_spring_velocity_init)

    def compute(self,
                state: ProsthesisState,
                ankle_torque_commanded: float,
                dt: float) -> tuple[float, float]:

        if not self._verified:
            raise NotImplementedError("Ankle Torque Controller not verified yet!")

        ankle_torque_commanded_dot = self._dd_tau.compute(ankle_torque_commanded, dt)
        ankle_torque_commanded_ddot = self._dd_tau_dot.compute(ankle_torque_commanded_dot, dt)

        ankle_spring_velocity_filtered = self._lowpass_spring_velocity.compute(state.ankle_spring_velocity)

        ankle_torque_error = ankle_torque_commanded - state.ankle_torque_measured
        ankle_velocity_desired = (ankle_spring_velocity_filtered
                                  + ankle_torque_commanded_dot / self.ankle_stiffness
                                  + self.ankle_kp * ankle_torque_error)

        ankle_motor_torque_ff_desired = (ankle_torque_commanded
                                         + (self.ankle_motor_damping / self.ankle_stiffness) * ankle_torque_commanded_dot
                                         + (self.ankle_motor_inertia / self.ankle_stiffness) * ankle_torque_commanded_ddot)

        return ankle_velocity_desired, ankle_motor_torque_ff_desired


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
class AnkleVelocitySaturation:
    def __init__(self, ankle: AnkleParams = params.ANKLE, common: CommonParams = params.COMMON):
        self.ankle_vel_max = ankle.vel_max
        self.ankle_torque_max = ankle.torque_max
        self.ankle_angle_min = ankle.angle_min
        self.ankle_angle_max = ankle.angle_max

        self._lowpass_velocity = Lowpass(
            cutoff_freq=ankle.vel_sat_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=ankle.vel_sat_lowpass_order,
        )
        self._lowpass_torque_ff = Lowpass(
            cutoff_freq=ankle.vel_sat_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=ankle.vel_sat_lowpass_order,
        )

        # change to True once everything is verified as safe to use.
        self._verified = True

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

        velocity_saturated = clamp(ankle_velocity_desired,
                                   velocity_lower, velocity_upper)
        torque_ff_saturated = clamp(ankle_motor_torque_ff_desired,
                                    torque_lower, torque_upper)

        ankle_velocity_commanded = self._lowpass_velocity.compute(velocity_saturated)
        ankle_motor_torque_ff = self._lowpass_torque_ff.compute(torque_ff_saturated)

        return ankle_velocity_commanded, ankle_motor_torque_ff
