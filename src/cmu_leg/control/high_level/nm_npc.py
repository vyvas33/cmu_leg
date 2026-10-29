"""Walking policy: neuromuscular stance + neural-predictive knee swing.

A sibling of NMMinJerk, not a replacement. Stance and the whole gait FSM are
the same; what changes is the knee during swing, which follows the neural
predictor through the three phases of `NN swing torque Control` instead of a
min-jerk trajectory.

MinJerkSwing still runs every swing. It owns the ankle outright -- the NN path
has nothing to say about it -- and it owns the knee too whenever the NN is not
armed, so a cold predictor or a stalled estimator falls back to the controller
we already fly.

Ownership, since it is split three ways:
    PhaseChart   which phase is driving
    Phase1/2/3   the knee angle command
    this class   the predictor's buffer, knee_desired and its derivatives, the
                 hip filters, the entry latches, the clamp and the pid
"""
import cmu_leg.config.params as params
from cmu_leg.config.params import CommonParams, NNSWING, NNSwingParams
from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.high_level.nm_minjerk import GaitPhase
from cmu_leg.control.high_level.walking.minjerk_swing import MinJerkSwing
from cmu_leg.control.high_level.walking.neuromuscular_stance import NeuromuscularStance
from cmu_leg.control.high_level.walking.nn_predictor import NeuralPredictor
from cmu_leg.control.high_level.walking.nn_swing import Phase, PhaseChart, \
    Phase1, Phase2, Phase3
from cmu_leg.control.utils import BackwardDifference, Lowpass, PID, clamp
from cmu_leg.structures.commands import TorqueDesired
from cmu_leg.structures.state import ProsthesisState

# Cutoffs the baseline applies to the hip signals inside `NN swing torque
# Control`, before the chart or any phase sees them. 2nd-order Butterworth.
HIP_ANGLE_CUTOFF = 50.0     # Hz
HIP_VEL_CUTOFF = 25.0       # Hz


class NMNPC(TorqueController):
    """Neuromuscular stance + neural-predictive swing, knee only."""

    _verified = False

    def __init__(self, cfg: NNSwingParams = NNSWING,
                 common: CommonParams = params.COMMON):
        self._cfg = cfg
        self._dt = dt = 1.0 / common.control_sample_rate
        self._stance_ctrl = NeuromuscularStance(common=common)
        self._swing_ctrl = MinJerkSwing()
        self._gait = GaitPhase(dt)

        self._predictor = NeuralPredictor(common=common)
        self._chart = PhaseChart(cfg)
        self._phase1 = Phase1(cfg, common)
        self._phase2 = Phase2(cfg, common)
        self._phase3 = Phase3(cfg, common)
        self._pid = PID(cfg.knee_kp, cfg.knee_ki, cfg.knee_kd,
                        cfg.deriv_filter_n)

        self._hip_angle_lp = Lowpass(HIP_ANGLE_CUTOFF,
                                     common.control_sample_rate, 2)
        self._hip_vel_lp = Lowpass(HIP_VEL_CUTOFF,
                                   common.control_sample_rate, 2)
        # knee_desired's first two rates, which phases 2 and 3 latch on entry
        self._knee_vel = BackwardDifference()
        self._knee_accel = BackwardDifference()

        self.knee_desired = 0.0
        self._vel = 0.0
        self._accel = 0.0
        self._t_prev = 0.0
        # the pid is not called in stance, nor while min-jerk holds the knee,
        # so its derivative filter goes stale across those gaps
        self._pid_gapped = True

    def reset(self, state: ProsthesisState, time: float) -> None:
        """Entered from the mode FSM, which hands over unloaded, so the gait
        FSM starts in swing. The predictor keeps whatever history it has;
        `ready` reports whether that is enough.
        """
        self._gait.reset()
        self._stance_ctrl.reset(state, time)
        self._swing_ctrl.reset(state, time, None)
        self._chart.reset()
        self._pid.reset()
        self._hip_angle_lp.reset(state.hip_angle)
        self._hip_vel_lp.reset(state.hip_velocity)
        self.knee_desired = state.knee_angle
        self._knee_vel.reset(self.knee_desired)
        self._knee_accel.reset(0.0)
        self._vel = self._accel = 0.0
        self._t_prev = time - self._dt
        self._pid_gapped = True

    @property
    def in_stance(self) -> bool:
        return self._gait.in_stance

    @property
    def last_stance_duration(self) -> float | None:
        return self._gait.last_stance_duration

    @property
    def phase(self) -> Phase:
        return self._chart.phase

    @property
    def armed(self) -> bool:
        """Whether the NN may drive the knee.

        The baseline gates on `computation_signal AND nn_enable`, where
        computation_signal latches on the foot leaving its stationary phase and
        clears at heel strike, after one full gait. Our equivalent is a warm
        buffer and a measured stance behind us: the predictor reads the buffer
        on demand, so the lead time the baseline needed to settle its network
        does not apply.
        """
        return (self._predictor.ready
                and self._gait.last_stance_duration is not None)

    def _compute(self, state: ProsthesisState, time: float) -> TorqueDesired:
        dt = time - self._t_prev
        self._t_prev = time

        # The predictor eats the COMMANDED knee, which is the measured angle
        # while loaded. Pushed before anything else runs, so the window ends on
        # this tick but still holds last tick's command, as the baseline's data
        # store does.
        if self._gait.in_stance:
            self.knee_desired = state.knee_angle
        self._predictor.update(state, self.knee_desired, time)

        hip_angle = self._hip_angle_lp.compute(state.hip_angle)
        hip_vel = self._hip_vel_lp.compute(state.hip_velocity)

        heel_strike, toe_off = self._gait.update(state.contact)
        if heel_strike:
            self._stance_ctrl.reset(state, time)
        if toe_off:
            self._swing_ctrl.reset(state, time,
                                   self._gait.last_stance_duration)

        if self._gait.in_stance:
            self._chart.reset()
            self._vel = self._accel = 0.0
            self._knee_vel.reset(self.knee_desired)
            self._knee_accel.reset(0.0)
            self._pid_gapped = True
            return self._stance_ctrl.compute(state, time)

        # min-jerk runs every swing tick: it owns the ankle, and its knee is
        # the fallback whenever the NN is not driving
        mj = self._swing_ctrl.compute(state, time)
        if not self.armed:
            self._chart.reset()
            self._pid_gapped = True
            return mj

        phase = self._chart.update(True, state.standing, self.knee_desired,
                                   hip_angle, state.hip_pos_x,
                                   self._phase1.descending)
        if phase is Phase.IDLE:
            self._pid_gapped = True
            return mj

        self._enter(phase, hip_vel, time)
        knee = self._advance(phase, time, hip_angle, hip_vel)

        self.knee_desired = knee
        self._vel = self._knee_vel.compute(knee, dt)
        self._accel = self._knee_accel.compute(self._vel, dt)

        error = clamp(knee, self._cfg.knee_min, self._cfg.knee_max) \
            - state.knee_angle
        if self._pid_gapped:
            # The baseline leaves the filter stale here, so its first swing
            # tick differences a fresh error against one from the previous
            # swing -- measured one-tick jumps of up to 62.7 Nm against a
            # 0.36 Nm typical step. Seeding costs nothing on every other tick.
            self._pid.resume(error)
            self._pid_gapped = False
        return TorqueDesired(knee_desired=self._pid.compute(error, dt),
                             ankle_desired=mj.ankle_desired)

    def _enter(self, phase: Phase, hip_vel: float, time: float) -> None:
        """Hand one phase its entry state, on the tick it takes over.

        The pid is NOT reset at swing entry, matching the baseline. Its
        reset port is driven by difference blocks that live inside
        `torque computation`, which is enabled by `swing` -- so they are
        frozen through stance and compute swing[now] - swing[last enabled
        tick] = 0 at entry. The enable gating defeats the reset logic, and
        the pid resumes with the derivative-filter and integrator state from
        the previous swing. Resetting instead costs 2.2 Nm rms against 0.6.
        """
        if not self._chart.entered:
            return
        if phase is Phase.ONE:
            self._phase1.reset(self.knee_desired, time)
        elif phase is Phase.TWO:
            self._phase2.reset(self.knee_desired, self._vel, hip_vel, time)
        elif phase is Phase.THREE:
            self._phase3.reset(self.knee_desired, self._vel, self._accel, time)

    def _advance(self, phase: Phase, time: float, hip_angle: float,
                 hip_vel: float) -> float:
        if phase is Phase.ONE:
            if self._phase1.needs_trajectory(time):
                self._phase1.set_trajectory(self._predictor.predict())
            return self._phase1.compute(time)
        if phase is Phase.TWO:
            return self._phase2.compute(time, hip_angle, hip_vel)
        return self._phase3.compute(time, hip_angle)
