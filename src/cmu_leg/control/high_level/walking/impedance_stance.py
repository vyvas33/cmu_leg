"""
Impedance Stance Controller
This controller is a finite state machine controller with three stance phases. 
The state machine lives here and the compute method ouputs stance torque directly based on the current phase.
"""

from cmu_leg.control.high_level.base_controller import TorqueController
import cmu_leg.config.params as params
from cmu_leg.config.params import CommonParams, ImpedanceParams
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired

class ImpedanceStance(TorqueController):
    """Three-phase impedance stance, joint-local (no hip/leg angle).

    Velocity is used RAW, deliberately. The baseline filters joint velocity in
    Standing Mode Control (25/20 Hz) and in Min Jerk Swing, but Stance Torques
    reads knee/ankle velocity straight off the bus with nothing in between --
    verified against every lowpass block in the model, 2026-08-10. Adding a
    filter here would be a divergence, not a port, and the phase lag it adds
    sits directly on the damping term.

    Damping is NOT floored here: params.IMPEDANCE already stores the clamped
    values, matching where the baseline applies its saturation (on load, in
    "Split up parameters"). Flooring again would be a second, hidden clamp.

    The k and b in params.IMPEDANCE are PER-KILOGRAM (Nm/rad/kg, Nm*s/rad/kg).
    The baseline multiplies the stance branch by stance_body_weight in
    "Stance//Swing Torque Mixing" -- only stance; the swing branch is unscaled.
    We fold that factor in here so this class, like every other
    TorqueController, returns torque in Nm.
    """

    def __init__(self, imp: ImpedanceParams = params.IMPEDANCE,
                 common: CommonParams = params.COMMON):
        self._phase = 0 # stance 0,1,2
        self._gain = (imp.stance1, imp.stance2, imp.stance3)

        self._thresh12 = imp.stance_12_threshold
        self._thresh23 = imp.stance_23_threshold
        self._body_weight = common.stance_body_weight

    def reset(self, state, time):
        self._phase = 0

    def _compute(self, state, time):
        if self._phase == 0 and state.knee_angle  > self._thresh12: self._phase = 1
        elif self._phase == 1 and state.ankle_angle > self._thresh23: self._phase = 2
        g = self._gain[self._phase]

        knee  = -g.knee_k *(state.knee_angle  - g.knee_off ) - g.knee_b *state.knee_velocity
        ankle = -g.ankle_k*(state.ankle_angle - g.ankle_off) - g.ankle_b*state.ankle_velocity

        return TorqueDesired(knee * self._body_weight, ankle * self._body_weight)




