from abc import ABC, abstractmethod
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired


class TorqueController(ABC):
    _verified: bool = True

    def compute(self, state: ProsthesisState, time: float) -> TorqueDesired:
        if not self._verified:
            raise NotImplementedError(f"{type(self).__name__} not verified yet!")
        return self._compute(state, time)

    @abstractmethod
    def _compute(self, state: ProsthesisState, time: float) -> TorqueDesired:
        """Return (knee_torque, ankle_torque) in N*m for the current state/time."""
        ...

    @abstractmethod
    def reset(self, state, time) -> None:
        ...
