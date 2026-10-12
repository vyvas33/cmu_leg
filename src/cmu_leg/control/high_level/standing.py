# standing controller

from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.utils import Lowpass
import cmu_leg.config.params as params
from cmu_leg.config.params import StandingParams, CommonParams
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.structures.commands import TorqueDesired


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

        knee_desired = self._min_jerk(self._knee_start, self._knee_target, self._t0, self._duration, time)
        ankle_desired = self._min_jerk(self._ankle_start, self._ankle_target, self._t0, self._duration, time)

        return TorqueDesired(
            knee_desired=self._pd(knee_desired, state.knee_angle, knee_velocity),
            ankle_desired=self._pd(ankle_desired, state.ankle_angle, ankle_velocity),
        )

    def _pd(self, angle_desired: float, angle: float, velocity: float) -> float:
        return self._kp * (angle_desired - angle) - self._kd * velocity

    def _min_jerk_factor(self, tau: float) -> float:
        """Quintic min-jerk blend s(tau) for tau in [0, 1].

        s = 10*tau^3 - 15*tau^4 + 6*tau^5, with s(0)=0, s(1)=1 and zero velocity
        and acceleration at both ends.
        """
        return tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)


    def _min_jerk(self, start: float, end: float, t0: float, duration: float, time: float) -> float:
        """Min-jerk position from `start` to `end` over `duration`, starting at `t0`.

        Holds at `start` before t0 and at `end` after t0 + duration (tau clamps).
        """
        tau = (time - t0) / duration  # [0, 1]
        tau = 0.0 if tau < 0.0 else 1.0 if tau > 1.0 else tau
        return start + (end - start) * self._min_jerk_factor(tau)
