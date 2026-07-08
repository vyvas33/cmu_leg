"""
Impedance Stance Controller
This controller is a finite state machine controller with three stance phases. 
The state machine lives here and the compute method ouputs stance torque directly based on the current phase.
"""

from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.utils import Lowpass
import cmu_leg.config.params as params
from cmu_leg.config.params import CommonParams, ImpedanceParams
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired

class ImpedanceStance(TorqueController):
    def __init__(self, imp: ImpedanceParams = params.IMPEDANCE):
        self._phase = 0 # stance 0,1,2
        self._gain = (imp.stance1, imp.stance2, imp.stance3)

        self._thresh12 = imp.stance_12_threshold
        self._thresh23 = imp.stance_23_threshold
        
        #initialise low pass filters for velocity?

    def reset(self, state, time):
        self._phase = 0
        #reset low pass filters 

    def _compute(self, state, time):
        if self._phase == 0 and state.knee_angle  > self._thresh12: self._phase = 1
        elif self._phase == 1 and state.ankle_angle > self._thresh23: self._phase = 2
        g = self._gain[self._phase]

        # call low pass filters here, pass for now
        knee_vel = state.knee_velocity
        ankle_vel = state.ankle_velocity

        knee  = -g.knee_k *(state.knee_angle  - g.knee_off ) - g.knee_b *knee_vel
        ankle = -g.ankle_k*(state.ankle_angle - g.ankle_off) - g.ankle_b*ankle_vel

        return TorqueDesired(knee, ankle)




