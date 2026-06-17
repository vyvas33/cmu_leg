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

from cmu_leg.types.state import ProsthesisState
from cmu_leg.control.utils import BackwardDifference
import cmu_leg.config.params as params
from cmu_leg.config.params import KneeParams, CommonParams

class KneeTorqueController:
    def __init__(self, knee: KneeParams = params.KNEE, common: CommonParams = params.COMMON):
        self.knee_kp = knee.kp
        self.knee_stiffness = knee.stiffness
        self.knee_motor_damping = knee.motor_damping
        self.knee_motor_inertia = knee.motor_inertia
        self.knee_motor_eff = knee.motor_eff
        self.sample_time = common.sample_time

        self._dd_tau = BackwardDifference(self.sample_time)
        self._dd_tau_dot = BackwardDifference(self.sample_time)

        # change to True once everything is verified as safe to use.
        self._verified = False

    # Reset the derivative state so the next compute() does not emit a step.
    def reset(self, knee_torque_commanded_init: float = 0.0) -> None:
        self._dd_tau.reset(knee_torque_commanded_init)
        self._dd_tau_dot.reset(0.0)

    def compute(self,
                state: ProsthesisState,
                knee_torque_commanded: float) -> tuple[float, float]:

        if not self._verified:
            raise NotImplementedError("Knee Torque Controller not verified yet!")

        knee_torque_commanded_dot = self._dd_tau.compute(knee_torque_commanded)
        knee_torque_commanded_ddot = self._dd_tau_dot.compute(knee_torque_commanded_dot)

        knee_torque_error = knee_torque_commanded - state.knee_torque_measured
        knee_velocity_desired = (state.knee_velocity
                                 + knee_torque_commanded_dot / self.knee_stiffness
                                 + self.knee_kp * knee_torque_error)

        knee_motor_torque_ff_desired = (knee_torque_commanded
                                        + (self.knee_motor_damping / self.knee_stiffness) * knee_torque_commanded_dot
                                        + (self.knee_motor_inertia / self.knee_stiffness) * knee_torque_commanded_ddot) / self.knee_motor_eff

        return knee_velocity_desired, knee_motor_torque_ff_desired
