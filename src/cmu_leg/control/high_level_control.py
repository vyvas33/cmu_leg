"""High-level control orchestrator: the mode FSM, the behavioural policies it
selects, and the single desired joint torque it emits.

Two levels of state machine, deliberately separated:

  MODE FSM   here. standing <-> walking <-> hold. Reads operator intent
             and `state.contact`, and owns which policy runs.
  GAIT FSM   inside NMMinJerk. stance <-> swing. Private -- `stance` is a
             control state and never goes on the bus (see structures/state.py).

They share exactly one signal, in one direction: `state.contact`, which the
proprioceptive estimator publishes as a measurement. Neither reaches into the
other. The baseline flattens both into a single Stateflow chart
(`Get Prosthesis State/State Machine`) and consequently has to publish `stance`
on its bus; splitting them is why we do not.

The mode FSM mirrors that chart's transitions, minus the terms that need the
IMU. See ModeParams for which guards were dropped and why the safety-critical
one was not.
"""

from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.high_level.standing import StandingController
from cmu_leg.control.high_level.nm_minjerk import NMMinJerk
import cmu_leg.config.params as params
from cmu_leg.config.params import ModeParams
from cmu_leg.runtime.operator import OperatorInput
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.commands import TorqueDesired, Mode


class ProsthesisControl:
    def __init__(self, cfg: ModeParams = params.MODE,
                 walking: TorqueController | None = None,
                 standing: TorqueController | None = None):
        self._cfg = cfg
        self._standing = standing if standing is not None else StandingController()
        # The walking policy is swappable so a bench run can put something
        # else in that slot (see control/high_level/bench_nn_phases.py)
        # without touching the mode FSM around it.
        self._walking = walking if walking is not None else NMMinJerk()

        self.mode = Mode.INITIALIZING
        self._last_seq = -1
        self._last_swing_seq = 0
        self._since_operator = 0.0
        self._enabled_for = 0.0     # continuous seconds of control_enabled

    # ------------------------------------------------------------------ mode

    def _still(self, state: ProsthesisState) -> bool:
        """Encoder-only stand-in for the baseline's |hip_velocity| < 0.1."""
        v = self._cfg.still_joint_velocity
        return abs(state.knee_velocity) < v and abs(state.ankle_velocity) < v

    def _next_mode(self, state: ProsthesisState, op: OperatorInput,
                   dt: float) -> Mode:
        # Dwell on a CONTINUOUS enable: a drive that bounces out of
        # OPERATION_ENABLED restarts the clock rather than banking credit.
        self._enabled_for = (self._enabled_for + dt) if state.control_enabled else 0.0
        # A source that has stopped updating is treated as gone. Fall to
        # HOLD -- never to walking, never hold-last.
        if op.seq == self._last_seq:
            self._since_operator += dt
        else:
            self._since_operator = 0.0
            self._last_seq = op.seq
        if self._since_operator > self._cfg.operator_timeout:
            return Mode.HOLD

        # Getting INTO hold is unconditional from anywhere; getting out of it
        # is guarded. That asymmetry is the point of the state, and it
        # matches the baseline's bare `[hold_toggle]` transition.
        if op.hold_toggle:
            return Mode.HOLD

        if self.mode is Mode.INITIALIZING:
            # Wait for the drives to actually reach OPERATION_ENABLED rather
            # than for a timeout that assumes they will have. This is the
            # baseline's `GetInitialState -> StandingMode [control_enabled &&
            # standing_mode_toggle]`, and it replaces the old
            # control_warmup_s >= low_warmup_s + actuation_warmup_margin_s
            # assertion -- a guess at how long the CiA-402 enable walk takes,
            # which passed silently if the walk ever ran long.
            if op.standing_toggle and self._enabled_for >= self._cfg.enable_dwell:
                return Mode.STANDING
            return Mode.INITIALIZING

        if self.mode is Mode.HOLD:
            if op.standing_toggle and self._still(state):
                return Mode.STANDING
            return Mode.HOLD

        if self.mode is Mode.STANDING:
            # The safety guard: walking is only ever entered UNLOADED, so the
            # gait FSM starts in swing and the knee stop cannot fire against a
            # straight standing knee. The baseline adds hip_velocity > 0 and
            # leg_angle > 90 deg to detect intent; those need the ESKF.
            if not op.standing_toggle and not state.contact:
                return Mode.WALKING
            return Mode.STANDING

        if self.mode is Mode.WALKING:
            # Baseline requires this from Stance specifically. We use `contact`
            # rather than reaching into the walking controller for `stance` --
            # same condition in substance, and it keeps the two FSMs sharing
            # only the one bus signal.
            if op.standing_toggle and state.contact and self._still(state):
                return Mode.STANDING
            return Mode.WALKING

        return self.mode

    # ------------------------------------------------------------ gait views

    # The gait FSM lives inside the walking policy and its phase is a control
    # state, not a measurement, so it never joins ProsthesisState. These are the
    # read-only path the run log uses to record it. Both are only MEANINGFUL
    # while mode is WALKING -- in any other mode the FSM is not being ticked and
    # holds whatever it last saw, which is why the log records the mode beside
    # them rather than these alone.
    @property
    def in_stance(self) -> bool:
        return self._walking.in_stance

    @property
    def last_stance_duration(self) -> float | None:
        return self._walking.last_stance_duration

    # --------------------------------------------------------------- compute

    def compute(self, state: ProsthesisState, time: float,
                op: OperatorInput, dt: float) -> TorqueDesired:
        previous = self.mode
        self.mode = self._next_mode(state, op, dt)

        # Bench swing trigger, edge-detected off the counter. Only meaningful
        # while walking, since that is the only mode running a swing.
        if op.swing_seq != self._last_swing_seq:
            self._last_swing_seq = op.swing_seq
            if self.mode is Mode.WALKING:
                self._walking.request_swing(op.swing_duration)

        if self.mode is not previous:
            # WALKING is left silent on purpose: entering it immediately solves
            # a swing, and that line already says where we are.
            if self.mode is not Mode.WALKING:
                print(f"[{time:7.2f}s] {self.mode.name}", flush=True)
            if self.mode is Mode.STANDING:
                self._standing.reset(state, time)
            elif self.mode is Mode.WALKING:
                self._walking.reset(state, time)

        if self.mode is Mode.STANDING:
            return self._standing.compute(state, time)
        if self.mode is Mode.WALKING:
            return self._walking.compute(state, time)
        return TorqueDesired()      # INITIALIZING and HOLD
