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

        # low pass filters 
        self._knee_vel_lowpass = Lowpass(
            cutoff_freq=standing.knee_vel_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=standing.vel_lowpass_order,
        )
        self._ankle_vel_lowpass = Lowpass(
            cutoff_freq=standing.ankle_vel_lowpass_cutoff,
            sample_rate=common.sample_rate,
            order=standing.vel_lowpass_order,
        )

        self._prev_standing = False
        self._knee_start = 0.0
        self._ankle_start = 0.0
        self._t0 = 0.0

    def reset(self, state: ProsthesisState, time: float) -> None:
        # reset the filters and set prev standing to false as a reset
        self._prev_standing = False
        self._knee_vel_lowpass.reset(state.knee_velocity)
        self._ankle_vel_lowpass.reset(state.ankle_velocity)

    def _compute(self, state: ProsthesisState, time: float) -> TorqueDesired:
        # always called, so the filters always stay warm
        knee_velocity = self._knee_vel_lowpass.compute(state.knee_velocity)
        ankle_velocity = self._ankle_vel_lowpass.compute(state.ankle_velocity)

        # at the rising edge, the start conditions are captured and stay the same as long as standing is True
        if state.standing and not self._prev_standing:
            self._knee_start = state.knee_angle
            self._ankle_start = state.ankle_angle
            self._t0 = time

        self._prev_standing = state.standing # after standing = true, we update prev standing every tick

        knee_desired = min_jerk(self._knee_start, self._knee_target, self._t0, self._duration, time)
        ankle_desired = min_jerk(self._ankle_start, self._ankle_target, self._t0, self._duration, time)

        return TorqueDesired(
            knee_desired=self._pd(knee_desired, state.knee_angle, knee_velocity),
            ankle_desired=self._pd(ankle_desired, state.ankle_angle, ankle_velocity),
        )

    def _pd(self, angle_desired: float, angle: float, velocity: float) -> float:
        return self._kp * (angle_desired - angle) - self._kd * velocity
