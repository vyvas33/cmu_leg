# standing controller

from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.utils import Lowpass
import cmu_leg.config.params as params
from cmu_leg.config.params import StandingParams, CommonParams
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired
from cmu_leg.control.high_level.minjerk import min_jerk


class StandingController(TorqueController):

    def __init__(self,
                 standing: StandingParams = params.STANDING,
                 common: CommonParams = params.COMMON):
        self._knee_target = standing.knee_target
        self._ankle_target = standing.ankle_target
        self._duration = standing.duration
        self._kp = standing.kp
        self._kd = standing.kd

        # Low-pass filters. These are FIXED-coefficient biquads, so they must be
        # designed at the rate they are actually ticked -- the control-process rate
        # (standing runs there), NOT the low-level sample_rate. Designing at the
        # wrong rate silently halves the effective cutoff.
        self._knee_vel_lowpass = Lowpass(
            cutoff_freq=standing.knee_vel_lowpass_cutoff,
            sample_rate=common.control_sample_rate,
            order=standing.vel_lowpass_order,
        )
        self._ankle_vel_lowpass = Lowpass(
            cutoff_freq=standing.ankle_vel_lowpass_cutoff,
            sample_rate=common.control_sample_rate,
            order=standing.vel_lowpass_order,
        )

        self._knee_start = 0.0
        self._ankle_start = 0.0
        self._t0 = 0.0

    def reset(self, state: ProsthesisState, time: float) -> None:
        # Called by the orchestrator on the standing rising edge. Latch the
        # min-jerk start conditions (current angles + start time) so the ramp
        # begins from where the leg actually is, and reseed the velocity filters
        # from the current velocities so the first output has no startup transient.
        self._knee_start = state.knee_angle
        self._ankle_start = state.ankle_angle
        self._t0 = time
        self._knee_vel_lowpass.reset(state.knee_velocity)
        self._ankle_vel_lowpass.reset(state.ankle_velocity)

    def _compute(self, state: ProsthesisState, time: float) -> TorqueDesired:
        # Start conditions were latched in reset() at the standing rising edge;
        # this just runs the min-jerk ramp + PD every tick.
        knee_velocity = self._knee_vel_lowpass.compute(state.knee_velocity)
        ankle_velocity = self._ankle_vel_lowpass.compute(state.ankle_velocity)

        knee_desired = min_jerk(self._knee_start, self._knee_target, self._t0, self._duration, time)
        ankle_desired = min_jerk(self._ankle_start, self._ankle_target, self._t0, self._duration, time)

        return TorqueDesired(
            knee_desired=self._pd(knee_desired, state.knee_angle, knee_velocity),
            ankle_desired=self._pd(ankle_desired, state.ankle_angle, ankle_velocity),
        )

    def _pd(self, angle_desired: float, angle: float, velocity: float) -> float:
        return self._kp * (angle_desired - angle) - self._kd * velocity
