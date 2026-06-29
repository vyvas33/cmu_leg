from cmu_leg.control.high_level.base_controller import TorqueController

class ImpedanceMinJerk(TorqueController):
    def __init__(self):
        pass

    def _compute(self, state, time):
        return super()._compute(state, time)
    
    def reset(self, state, time):
        return super().reset(state, time)

