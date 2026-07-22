"""Knee low-level pipeline: conditioning -> torque controller -> velocity
saturation. Three independent stages, each with its own `_verified` gate and
reset, matching the nm_nn_swing Knee subsystems. Kept as separate classes in one
file so the per-stage verification/reset boundaries stay intact.
"""

import cmu_leg.config.params as params
from cmu_leg.config.params import KneeParams, CommonParams
from cmu_leg.control.utils import Lowpass, BackwardDifference, clamp
from cmu_leg.types.commands import TorqueDesired
from cmu_leg.types.state import ProsthesisState


# This class takes in the state and the desired torque from high level controller, computes the commanded torque and sends it to knee torque controller.
class KneeTorqueConditioning:
    def __init__(self, knee: KneeParams = params.KNEE, common: CommonParams = params.COMMON):
        self.knee_torque_max = knee.torque_max
        self.knee_angle_max_soft = knee.angle_max_soft
        self.knee_angle_min_soft = knee.angle_min_soft
        self.c_jointstop = knee.c_jointstop
        self.w_max_jointstop = knee.w_max_jointstop
        self.damping_gain = common.damping_gain

        # Per-stop sign + limit, verified against nm_nn_swing Knee Torque
        # Conditioning (blk_16825). The two stop subsystems are NOT identical:
        #   Flexion Stop (blk_16860):   ext=+1, phiUpLimit=knee_angle_max_soft
        #   Extension Stop (blk_16837): ext=-1, phiUpLimit=knee_angle_min_soft
        # Same pattern as AnkleTorqueConditioning.
        self.flex_ext = knee.flex_ext      # +1, flexion stop at angle_max_soft
        self.extend_ext = knee.extend_ext  # -1, extension stop at angle_min_soft
        self._lowpass = Lowpass()

        # change this to true once everything is verified as safe to use.
        self._verified = True

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
        e_knee_position = self.flex_ext * self.c_jointstop * (knee_position - self.knee_angle_max_soft)

        e_knee_velocity = - self.flex_ext * (1/self.w_max_jointstop) * knee_velocity

        if e_knee_position > 0 and e_knee_velocity < 1:
            stop_torque = (-1) * e_knee_position * (1 - e_knee_velocity)
        else:
            stop_torque = 0.0

        return self.flex_ext * stop_torque

    def _extension_stop(self, knee_motor_position, knee_motor_velocity):
        e_knee_position = self.extend_ext * self.c_jointstop * (knee_motor_position - self.knee_angle_min_soft)

        e_knee_velocity = - self.extend_ext * (1/self.w_max_jointstop) * knee_motor_velocity

        if e_knee_position > 0 and e_knee_velocity < 1:
            stop_torque = (-1) * e_knee_position * (1 - e_knee_velocity)
        else:
            stop_torque = 0.0

        return self.extend_ext * stop_torque

    def _torque_saturation(self, torque_input, lo, up):

        return clamp(torque_input, lo, up)


# Knee torque controller. Reads from prosthesis state and the commanded torque
# coming out of conditioning. Produces:
#   - desired motor velocity
#   - torque feed-forward
#
# Equations (from baseline_impedance_min_jerk, Knee Torque Controller block):
#   tau_dot   = d/dt  tau_cmd                 # backward difference
#   tau_ddot  = d/dt  tau_dot                 # backward difference
#   omega_des = omega_knee + tau_dot / Ks + Kp * (tau_cmd - tau_meas)
#   tau_ff    = (tau_cmd + (bm/Ks)*tau_dot + (Jm/Ks)*tau_ddot) / eta_m
#
# The two derivative-of-velocity branches in the Simulink block are zero-gain
# and therefore omitted here.
class KneeTorqueController:
    def __init__(self, knee: KneeParams = params.KNEE, common: CommonParams = params.COMMON):
        self.knee_kp = knee.kp
        self.knee_stiffness = knee.stiffness
        self.knee_motor_damping = knee.motor_damping
        self.knee_motor_inertia = knee.motor_inertia
        self.knee_motor_eff = knee.motor_eff

        # Both differences run on the caller's measured dt. These two are the
        # most dt-sensitive quantities in the stack: tau_ddot is a difference OF
        # tau_dot, so a bad dt is compounded through both stages rather than
        # just added, and the result feeds the drive feedforward.
        self._dd_tau = BackwardDifference()
        self._dd_tau_dot = BackwardDifference()

        # change to True once everything is verified as safe to use.
        self._verified = True

    # Reset the derivative state so the next compute() does not emit a step.
    def reset(self, knee_torque_commanded_init: float = 0.0) -> None:
        self._dd_tau.reset(knee_torque_commanded_init)
        self._dd_tau_dot.reset(0.0)

    def compute(self,
                state: ProsthesisState,
                knee_torque_commanded: float,
                dt: float) -> tuple[float, float]:

        if not self._verified:
            raise NotImplementedError("Knee Torque Controller not verified yet!")

        knee_torque_commanded_dot = self._dd_tau.compute(knee_torque_commanded, dt)
        knee_torque_commanded_ddot = self._dd_tau_dot.compute(knee_torque_commanded_dot, dt)

        knee_torque_error = knee_torque_commanded - state.knee_torque_measured
        knee_velocity_desired = (state.knee_velocity
                                 + knee_torque_commanded_dot / self.knee_stiffness
                                 + self.knee_kp * knee_torque_error)

        knee_motor_torque_ff_desired = (knee_torque_commanded
                                        + (self.knee_motor_damping / self.knee_stiffness) * knee_torque_commanded_dot
                                        + (self.knee_motor_inertia / self.knee_stiffness) * knee_torque_commanded_ddot) / self.knee_motor_eff

        return knee_velocity_desired, knee_motor_torque_ff_desired


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
        self._verified = True

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

        velocity_saturated = clamp(knee_velocity_desired,
                                   velocity_lower, velocity_upper)
        torque_ff_saturated = clamp(knee_motor_torque_ff_desired,
                                    torque_lower, torque_upper)

        knee_velocity_commanded = self._lowpass_velocity.compute(velocity_saturated)
        knee_motor_torque_ff = self._lowpass_torque_ff.compute(torque_ff_saturated)

        return knee_velocity_commanded, knee_motor_torque_ff
