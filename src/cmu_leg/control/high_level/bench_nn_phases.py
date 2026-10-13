"""Bench policy: hold the knee, and run phases 2 and 3 while the foot is up.

For a worn leg with no load on it. Park the knee, lift the thigh, and the
shank should hold a fixed angle in the world while the knee flexes and extends
to keep it there. Phase 1 and the predictor are not involved, so this needs
neither the network nor hip position -- only hip angle and hip velocity.

The handover arms itself. Standing with the thigh down the target
`hip_angle - hip_knee_offset` sits below the knee, so phase 2 eases the knee
onto its lower clamp and waits. Lifting the thigh past about 22 deg raises the
target above the knee, which is phase 3's entry condition, and it takes over
with near-zero error -- so its deadbeat velocity term never sees a large one.

Gains are softened from NNSWING: knee_kd 15 -> 5, which cuts the handover's
derivative kick from 119 Nm to 48 and leaves the command identical. The pid is
also reset on every phase change and its integral is bounded -- neither of
which the baseline does, because the baseline never parks.

    ProsthesisControl(walking=BenchNNPhases(), standing=bench_standing())

Arm from a BENT knee. Standing's usual 5 deg leaves phase 2 only ~3 deg of
travel before it reaches the clamp, so the lift tests phase 3 and little else.
bench_standing() parks at BENCH_KNEE_TARGET instead, giving phase 2 a real gap
to close -- the slope ramp and the fade are then both exercised.
"""
from dataclasses import replace

import cmu_leg.config.params as params
from cmu_leg.config.params import CommonParams, NNSWING, STANDING, StandingParams
from cmu_leg.control.high_level.base_controller import TorqueController
from cmu_leg.control.high_level.standing import StandingController
from cmu_leg.control.high_level.walking.nn_swing import Phase2, Phase3
from cmu_leg.control.utils import BackwardDifference, Lowpass, PID, clamp
from cmu_leg.structures.commands import TorqueDesired
from cmu_leg.structures.state import ProsthesisState

BENCH = replace(NNSWING, knee_kd=5.0)

# Bound on the pid's integral term [Nm]. The baseline has no anti-windup and
# never needs one, because it only ever runs mid-swing. This policy parks on a
# frozen setpoint between lifts, where the knee sags against it and leaves a
# steady error -- at ki = 100 that is 5 Nm per second of windup, unbounded.
# The integral is what actually holds the knee, so it is bounded rather than
# removed. Comfortably above the 17 Nm peak the whole gesture needs.
I_LIMIT = 10.0

# Knee angle to park at before the lift, so phase 2 has something to do.
# Phase 2's slope ramp covers ~30 deg whatever the target, so parking shallower
# than that runs the command through the 2 deg clamp and pins the knee on its
# extension stop for the rest of the ramp.
BENCH_KNEE_TARGET = 30.0 * 3.141592653589793 / 180.0


def bench_standing(cfg: StandingParams = STANDING) -> StandingController:
    """StandingController parked at BENCH_KNEE_TARGET instead of 5 deg."""
    return StandingController(standing=replace(cfg, knee_target=BENCH_KNEE_TARGET))

HIP_ANGLE_CUTOFF = 50.0     # Hz, as the baseline filters it
HIP_VEL_CUTOFF = 25.0

HOLD, TWO, THREE = 0, 2, 3


class BenchNNPhases(TorqueController):
    """Knee held when loaded, phases 2 then 3 when the foot is up."""

    _verified = False

    def __init__(self, cfg=BENCH, common: CommonParams = params.COMMON):
        self._cfg = cfg
        self._dt = 1.0 / common.control_sample_rate
        self._p2 = Phase2(cfg, common)
        self._p3 = Phase3(cfg, common)
        self._pid = PID(cfg.knee_kp, cfg.knee_ki, cfg.knee_kd,
                        cfg.deriv_filter_n, i_limit=I_LIMIT)
        self._hip_lp = Lowpass(HIP_ANGLE_CUTOFF, common.control_sample_rate, 2)
        self._hv_lp = Lowpass(HIP_VEL_CUTOFF, common.control_sample_rate, 2)
        self._d_vel = BackwardDifference()
        self._d_acc = BackwardDifference()
        self.phase = HOLD
        self.knee_desired = 0.0
        self._vel = self._accel = 0.0
        self._t_prev = 0.0

    def reset(self, state: ProsthesisState, time: float) -> None:
        """Park on the current knee angle. Returns None."""
        self.phase = HOLD
        self.knee_desired = state.knee_angle
        self._pid.reset()
        self._hip_lp.reset(state.hip_angle)
        self._hv_lp.reset(state.hip_velocity)
        self._d_vel.reset(self.knee_desired)
        self._d_acc.reset(0.0)
        self._vel = self._accel = 0.0
        self._t_prev = time - self._dt

    def _compute(self, state: ProsthesisState, time: float) -> TorqueDesired:
        cfg = self._cfg
        dt = time - self._t_prev
        self._t_prev = time
        hip = self._hip_lp.compute(state.hip_angle)
        hip_vel = self._hv_lp.compute(state.hip_velocity)

        if state.contact:
            # Foot down: freeze the command where it is and let the pid hold
            # it. Not Mode.HOLD, which de-actuates the whole leg.
            if self.phase is not HOLD:
                self.phase = HOLD
                self.knee_desired = state.knee_angle
                self._vel = self._accel = 0.0
                self._pid.reset(self._error(state))
        elif self.phase is HOLD:
            self.phase = TWO
            self._p2.reset(self.knee_desired, 0.0, hip_vel, time)
            self._pid.reset(self._error(state))
        elif self.phase is TWO:
            if self.knee_desired < hip - cfg.hip_knee_offset and hip > 0.0:
                self.phase = THREE
                self._p3.reset(self.knee_desired, self._vel, self._accel, time)
                self._pid.reset(self._error(state))

        if self.phase is TWO:
            self.knee_desired = self._p2.compute(time, hip, hip_vel)
        elif self.phase is THREE:
            self.knee_desired = self._p3.compute(time, hip)

        self._vel = self._d_vel.compute(self.knee_desired, dt)
        self._accel = self._d_acc.compute(self._vel, dt)

        return TorqueDesired(knee_desired=self._pid.compute(self._error(state), dt),
                             ankle_desired=0.0)

    def _error(self, state: ProsthesisState) -> float:
        """Clamped command minus measured knee angle [rad]."""
        return clamp(self.knee_desired, self._cfg.knee_min,
                     self._cfg.knee_max) - state.knee_angle
