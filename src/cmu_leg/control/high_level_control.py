"""High-level control orchestrator: holds the behavioral policies, runs the
active one each tick, and emits a single desired joint torque for low level.

For now there is only the standing policy. Walking will join as a second
`TorqueController` slot, selected on the same mode flag.
"""

from cmu_leg.control.high_level.standing import StandingController
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired


class ProsthesisControl:
    def __init__(self):
        # policies
        self._standing = StandingController()
        self._prev_standing = False

    def compute(self, state: ProsthesisState, time: float) -> TorqueDesired:
        rising_edge = state.standing and not self._prev_standing
        self._prev_standing = state.standing

        if rising_edge:
            self._standing.reset(state, time)

        if state.standing:
            return self._standing.compute(state, time)

        return TorqueDesired()
