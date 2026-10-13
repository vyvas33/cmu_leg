from abc import ABC, abstractmethod
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.commands import TorqueDesired


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

    # The rest of what ProsthesisControl asks of a policy. Defaulted rather
    # than abstract: they serve the run log and the scripted operator, and a
    # policy with no gait FSM has no answer to give. Overriding is how a
    # policy that does opts in.
    @property
    def in_stance(self) -> bool:
        """Whether the policy considers the foot loaded. Logging only."""
        return False

    @property
    def last_stance_duration(self) -> float | None:
        """Duration [s] of the last completed stance, None if there is none."""
        return None

    def request_swing(self, duration: float) -> None:
        """Ask for one swing of `duration` [s]. Ignored where unsupported."""
