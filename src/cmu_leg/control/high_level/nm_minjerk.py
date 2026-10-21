"""Walking policy: neuromuscular stance + min-jerk swing.

Sits parallel to StandingController as a policy the orchestrator selects on the
mode flag. Everything gait-phase lives in here and nothing about it goes on the
bus -- `stance` is derived, used, and discarded within this class (see the note
in structures/state.py).

Three things this owns:
  * the gait FSM (stance <-> swing) over the estimator's `contact`
  * the phase edges that re-arm each controller
  * selecting stance or swing torque, and the stance duration that sizes swing

`contact` arrives already thresholded on the bus, from ProprioceptiveEstimator.
The muscle model and the swing trajectory live in NeuromuscularStance and
MinJerkSwing, which this holds.
"""

from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.high_level.walking.neuromuscular_stance import NeuromuscularStance
from cmu_leg.control.high_level.walking.minjerk_swing import MinJerkSwing
import cmu_leg.config.params as params
from cmu_leg.config.params import CommonParams, ContactParams
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.commands import TorqueDesired


class GaitPhase:
    """stance <-> swing from the estimator's `contact`, with a refractory hold.

    Reads the thresholded contact measurement off the bus; the threshold and
    the live zero belong to ProprioceptiveEstimator. What lives here is the
    FSM: when a contact change is allowed to become a phase change.

    De-chattering is a refractory hold rather than a debounce: a debounce would
    delay the heel-strike edge too and clip stance duty, whereas a hold only
    suppresses spurious re-triggers. It stands in for the baseline's leg-angle
    guard, which needs the IMU. See the note on ContactParams.min_stance.

    Also measures how long the last stance lasted, which is what sets the
    swing duration.
    """

    def __init__(self, dt: float, cfg: ContactParams = params.CONTACT):
        self._dt = dt
        self._cfg = cfg
        self.in_stance = False
        self._held = 0.0
        self.last_stance_duration = None    # s; None until one has completed

    def reset(self) -> None:
        """Enter in swing. The mode FSM only hands over while unloaded, so the
        first transition is a genuine heel strike."""
        self.in_stance = False
        self._held = 0.0
        self.last_stance_duration = None

    def update(self, contact: bool) -> tuple[bool, bool]:
        """Advance the phase. Returns (heel_strike, toe_off).

        A tick has three outcomes -- no transition, swing -> stance, or
        stance -> swing -- and each phase controller re-arms on its own entry
        edge, so both are reported. At most one is ever true.

        On toe_off, `last_stance_duration` is already updated: it is assigned
        from `_held` before `_held` is cleared, so a caller may read it in the
        same tick to size the swing.
        """
        cfg = self._cfg
        self._held += self._dt
        floor = cfg.min_stance if self.in_stance else cfg.min_swing

        heel_strike = False
        toe_off = False
        if contact != self.in_stance and self._held >= floor:
            if not contact:                      # toe off: stance just ended
                self.last_stance_duration = self._held
                toe_off = True
            else:
                heel_strike = True
            self.in_stance = contact
            self._held = 0.0
        return heel_strike, toe_off


class NMMinJerk(TorqueController):
    """Neuromuscular stance + min-jerk swing, selected by the gait FSM."""

    _verified = True

    def __init__(self, common: CommonParams = params.COMMON):
        dt = 1.0 / common.control_sample_rate
        self._stance_ctrl = NeuromuscularStance(common=common)
        self._swing_ctrl = MinJerkSwing()
        self._gait = GaitPhase(dt)
        self._swing_request = None      # bench trigger; see request_swing()

    def reset(self, state, time) -> None:
        """Entered from the mode FSM, which only hands over while unloaded, so
        the gait FSM starts in swing and the first transition is a real heel
        strike.

        Swing is reset with no stance duration, since none has been measured
        yet; it takes the maximum swing duration and runs a trajectory to the
        extend pose, parking the leg ready for first contact.
        """
        self._gait.reset()
        self._swing_request = None
        self._stance_ctrl.reset(state, time)
        self._start_swing(state, time, None)

    # Read-only views of the gait FSM, for the run log. `stance` stays a control
    # state and does NOT go on the bus (see the note on ProsthesisState.contact);
    # exposing it here lets the logger see the FSM output without the runtime
    # reaching into `_gait`, and without another consumer being able to write it.
    @property
    def in_stance(self) -> bool:
        return self._gait.in_stance

    @property
    def last_stance_duration(self) -> float | None:
        return self._gait.last_stance_duration

    def request_swing(self, duration: float) -> None:
        """Ask for one swing of `duration` seconds, served on the next tick.

        For a leg hanging in a harness: the foot never loads, so contact never
        asserts and the gait FSM produces no toe-off to solve a trajectory from.
        The request is expressed as the stance duration that would have sized
        this swing, so it goes through the same path as a real toe-off and the
        duration clamp still applies.
        """
        self._swing_request = duration / params.MJSWING.swing_to_stance_ratio

    def _start_swing(self, state, time, stance_duration) -> None:
        """Solve a swing and say so. The line is the only live trace of the gait
        FSM during a run; everything else goes to the run-dir log at teardown."""
        self._swing_ctrl.reset(state, time, stance_duration)
        print(f"[{time:7.2f}s] swing   T={self._swing_ctrl._T:.2f}s", flush=True)

    def _compute(self, state, time) -> TorqueDesired:
        heel_strike, toe_off = self._gait.update(state.contact)

        # A pending bench request stands in for a toe-off. Only honoured while
        # the gait FSM is already in swing, so it cannot cut a real stance short.
        if self._swing_request is not None and not self._gait.in_stance:
            if not toe_off:
                self._start_swing(state, time, self._swing_request)
            self._swing_request = None

        # Each controller re-arms on its own entry edge; neither resets on exit.
        # On heel strike NeuromuscularStance seeds l_ce at equilibrium, prefills
        # its afferent buffers, and arms its knee stop on time-since-reset. On
        # toe off swing solves its trajectory from the joint state here and
        # sizes it from last_stance_duration, which is current on this tick.
        if heel_strike:
            self._stance_ctrl.reset(state, time)
            print(f"[{time:7.2f}s] stance", flush=True)
        if toe_off:
            self._start_swing(state, time, self._gait.last_stance_duration)

        if self._gait.in_stance:
            return self._stance_ctrl.compute(state, time)
        return self._swing_ctrl.compute(state, time)
