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


from cmu_leg.types.state import ProsthesisState
from cmu_leg.control.utils import BackwardDifference, Lowpass
import cmu_leg.config.params as params
from cmu_leg.config.params import AnkleParams, CommonParams


class AnkleTorqueController:
    def __init__(self, ankle: AnkleParams = params.ANKLE, common: CommonParams = params.COMMON):
        self.ankle_kp = ankle.kp
        self.ankle_stiffness = ankle.stiffness
        self.ankle_motor_damping = ankle.motor_damping
        self.ankle_motor_inertia = ankle.motor_inertia
        self.sample_time = common.sample_time

        self._dd_tau = BackwardDifference(self.sample_time)
        self._dd_tau_dot = BackwardDifference(self.sample_time)
        self._lowpass_spring_velocity = Lowpass(
            cutoff_freq=ankle.torque_ctrl_spring_vel_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=ankle.torque_ctrl_spring_vel_lowpass_order,
        )

        # change to True once everything is verified as safe to use.
        self._verified = False

    # Reset derivative + lowpass state so the next compute() does not emit a step.
    def reset(self,
              ankle_torque_commanded_init: float = 0.0,
              ankle_spring_velocity_init: float = 0.0) -> None:
        self._dd_tau.reset(ankle_torque_commanded_init)
        self._dd_tau_dot.reset(0.0)
        self._lowpass_spring_velocity.reset(ankle_spring_velocity_init)

    def compute(self,
                state: ProsthesisState,
                ankle_torque_commanded: float) -> tuple[float, float]:

        if not self._verified:
            raise NotImplementedError("Ankle Torque Controller not verified yet!")

        ankle_torque_commanded_dot = self._dd_tau.compute(ankle_torque_commanded)
        ankle_torque_commanded_ddot = self._dd_tau_dot.compute(ankle_torque_commanded_dot)

        ankle_spring_velocity_filtered = self._lowpass_spring_velocity.compute(state.ankle_spring_velocity)

        ankle_torque_error = ankle_torque_commanded - state.ankle_torque_measured
        ankle_velocity_desired = (ankle_spring_velocity_filtered
                                  + ankle_torque_commanded_dot / self.ankle_stiffness
                                  + self.ankle_kp * ankle_torque_error)

        ankle_motor_torque_ff_desired = (ankle_torque_commanded
                                         + (self.ankle_motor_damping / self.ankle_stiffness) * ankle_torque_commanded_dot
                                         + (self.ankle_motor_inertia / self.ankle_stiffness) * ankle_torque_commanded_ddot)

        return ankle_velocity_desired, ankle_motor_torque_ff_desired
