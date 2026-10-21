"""NN swing control: three phases and the chart that sequences them.

Ported from `NN swing torque Control` in nm_nn_swing. Phase 1 tracks the
network's predicted knee trajectory, phase 2 extends the knee onto a landing
pose, and phase 3 holds it there until contact.

Phase1 does not hold the predictor. It asks for a trajectory when it wants one
and takes it in, so the caller chooses the source:

    if phase1.needs_trajectory(t):
        phase1.set_trajectory(predictor.predict())
    setpoint = phase1.compute(t)
"""
from enum import IntEnum

import numpy as np

from cmu_leg.config.params import COMMON, NNSWING, CommonParams, NNSwingParams
from cmu_leg.control.high_level.walking.nn_predictor import PLAYBACK_DT
from cmu_leg.config.params import LEG
from cmu_leg.control.utils import clamp
from cmu_leg.estimation.leg_kinematics import (HEEL_ANGLE, HEEL_RADIUS,
                                               heel_forward)


class Phase1:
    def __init__(self, cfg: NNSwingParams = NNSWING,
                 common: CommonParams = COMMON):
        self._cfg = cfg
        self._dt = 1.0 / common.control_sample_rate
        self._n = None                  # samples in a latched trajectory
        self._grid = None               # their times from the latch [s]
        self._traj = np.zeros(0)        # latched, re-anchored, leg frame [rad]

        self.knee_desired = 0.0         # last command [rad], also the anchor
        self.descending = False         # see _note_descent
        self._descends = False          # the current trajectory, on its own
        self._t_start = 0.0             # when the current trajectory started
        self._t_next_start = 0.0        # when the next one starts
        self._pending = False           # period elapsed, swap next tick

    def reset(self, knee_angle: float, t: float) -> None:
        """Enter a swing. knee_angle [rad] anchors the first trajectory.

        Leaves needs_trajectory() true, so the caller sets one this tick.
        """
        self.knee_desired = knee_angle
        self.descending = False
        self._descends = False
        self._traj = np.zeros(0)
        self._t_start = self._t_next_start = t
        self._pending = True

    def needs_trajectory(self, t: float) -> bool:
        """Whether to hand set_trajectory() a new one this tick.

        True on the first tick of a swing, then on the first tick after each
        trajectory_period. Call once per tick, before compute(): this is what
        notices that the period has elapsed.
        """
        if self._pending:
            return True

        elapsed = t - self._t_start
        if elapsed > self._cfg.trajectory_period - 1.5 * self._dt:
            # Notice it a tick early and swap on the next one, so the new
            # trajectory starts on the tick its clock reads 0 and the command
            # stays continuous. Holding the time in _t_next_start rather than
            # applying it here keeps this tick's clock on the old trajectory.
            self._t_next_start = t + self._dt
            self._pending = True
        return False

    def set_trajectory(self, traj: np.ndarray) -> None:
        """Take a new trajectory, shifted to continue from the last command.

        traj is absolute knee angles [rad] in our leg's frame, PLAYBACK_DT
        apart, as the predictor returns them. Only its shape is kept: the
        network sets how the knee moves from here, the current command sets
        where it starts.
        """
        if self._n is None:
            self._n = len(traj)
            self._grid = np.arange(self._n) * PLAYBACK_DT
        self._descends = bool(traj[-1] < traj[0])
        self._traj = traj - traj[0] + (self.knee_desired + traj[1] - traj[0])
        self._t_start = self._t_next_start
        self._pending = False

    def compute(self, t: float) -> float:
        """Knee angle command [rad] for this tick, held past the horizon.

        Also updates `descending`, which latches once a trajectory that turns
        back toward extension has been played for descend_hold. Latched, not
        momentary: it stays set until the next reset, so the phase cannot be
        pulled back by a later prediction that happens to rise.
        """
        if not len(self._traj):
            return self.knee_desired
        clock = t - self._t_start
        if self._descends and clock > self._cfg.descend_hold:
            self.descending = True
        self.knee_desired = float(
            np.interp(clock, self._grid, self._traj))
        return self.knee_desired


class Phase2:
    """Extend the knee onto its landing pose.

    From Xing's eq 4.3-4.4. The knee is driven toward
    `hip_angle - hip_knee_offset`, the angle at which the shank holds a fixed
    forward lean in the world frame, ready for touchdown.

    Velocity is `slope * hip_vel`. Slope ramps from whatever phase 1 left to
    slope_target; after the ramp it freezes and the velocity fades away with
    the error that is left.

        phase2.reset(knee_angle, knee_vel, hip_vel, t)
        setpoint = phase2.compute(t, hip_angle, hip_vel)
    """

    def __init__(self, cfg: NNSwingParams = NNSWING,
                 common: CommonParams = COMMON):
        self._cfg = cfg
        self._dt_nominal = 1.0 / common.control_sample_rate
        self.knee_desired = 0.0
        self._t_entry = 0.0
        self._t_prev = 0.0
        self._slope_ini = 0.0
        self._hip_vel_ini = 0.0
        self._step = 0.0          # slope increment per baseline tick
        self._n_steps = 0         # ramp length, in baseline ticks
        # state at the end of the ramp, which the fade is measured against
        self._knee_frozen = 0.0
        self._hip_frozen = 0.0
        self._hip_vel_frozen = 0.0

    def reset(self, knee_angle: float, knee_vel: float, hip_vel: float,
              t: float) -> None:
        """Take over from phase 1.

        knee_angle [rad] and knee_vel [rad/s] are phase 1's last command and
        its rate, hip_vel [rad/s] is the hip velocity now, t [s] the entry
        time. Returns None.
        """
        cfg = self._cfg
        self.knee_desired = knee_angle
        self._t_entry = t
        self._t_prev = t - self._dt_nominal
        self._hip_vel_ini = hip_vel
        # A standing hip gives no ratio. Fall back to 0, not slope_target:
        # a standstill means the knee is not moving relative to the hip, and
        # it leaves the ramp full length instead of jumping straight to the
        # saturated slope and the 3 rad/s floor. The baseline never had to
        # choose -- it only enters mid-swing, where its Divide would otherwise
        # produce Inf.
        self._slope_ini = knee_vel / hip_vel if abs(hip_vel) > 1e-6 else 0.0

        down = self._slope_ini > cfg.slope_target
        self._step = -cfg.slope_step if down else cfg.slope_step_up
        self._n_steps = int((cfg.slope_target - self._slope_ini) / self._step)

        self._knee_frozen = knee_angle
        self._hip_frozen = cfg.hip_angle_min
        self._hip_vel_frozen = hip_vel

    def compute(self, t: float, hip_angle: float, hip_vel: float) -> float:
        """Knee angle command [rad] for this tick.

        hip_angle [rad] and hip_vel [rad/s] are the measured hip. hip_angle is
        clamped below at hip_angle_min before use.
        """
        cfg = self._cfg
        hip_angle = max(hip_angle, cfg.hip_angle_min)
        dt = t - self._t_prev
        self._t_prev = t

        # the ramp advances on the baseline's tick, so elapsed time indexes it
        # the same way whatever rate we run at
        k = round((t - self._t_entry) / cfg.baseline_dt)
        ramping = k <= self._n_steps

        if ramping:
            frac = k / self._n_steps if self._n_steps else 1.0
            floor = self._hip_vel_ini + frac * (cfg.hip_vel_target
                                                - self._hip_vel_ini)
            hip_vel = max(hip_vel, floor)
            slope = self._slope_ini + self._step * k
        else:
            hip_vel = max(hip_vel, self._hip_vel_frozen)
            # the ramp's last grid value, which a slower tick could step over
            slope = self._slope_ini + self._step * self._n_steps

        knee_vel = slope * hip_vel
        if not ramping:
            left = self.knee_desired - hip_angle + cfg.hip_knee_offset
            at_freeze = abs(self._knee_frozen - self._hip_frozen
                            + cfg.hip_knee_offset)
            # The baseline takes abs(left), so once the knee passes below the
            # target the fade grows again and the command diverges -- measured
            # at -65 deg and at -inf on the two swings where phase 3 took over.
            # Flooring at zero stalls the knee on the target instead, so a
            # missed handover costs a held knee rather than a runaway. Only the
            # floor differs from the baseline; above the target the law is
            # untouched, including the case where a moving target pushes the
            # ratio past 1 and the knee re-accelerates.
            fade = max(left, 0.0) / at_freeze if at_freeze else 0.0
            knee_vel *= fade ** cfg.decay_exponent

        self.knee_desired += knee_vel * dt

        if ramping:
            # frozen from the tick the ramp ends on, after the knee advances
            self._knee_frozen = self.knee_desired
            self._hip_frozen = hip_angle
            self._hip_vel_frozen = hip_vel
        return self.knee_desired


class Phase3:
    """Hold the knee on its landing pose until contact.

    Xing's eq 4.5: the knee tracks `hip_angle - hip_knee_offset`, so it opens
    at whatever rate keeps the shank's world angle fixed. Phase 2 can only
    extend and cannot recover if it passes the target; this tracks it from
    either side.

    The command blends out of phase 2's entry velocity and acceleration over
    the first ~100 ms rather than snapping, which is eq 4.6.

        phase3.reset(knee_angle, knee_vel, knee_accel, t)
        setpoint = phase3.compute(t, hip_angle)
    """

    def __init__(self, cfg: NNSwingParams = NNSWING,
                 common: CommonParams = COMMON):
        self._cfg = cfg
        self._dt_nominal = 1.0 / common.control_sample_rate
        self.knee_desired = 0.0
        self.gamma = 1.0            # weight still on the entry velocity
        self._t_entry = 0.0
        self._t_prev = 0.0
        self._knee_vel_ini = 0.0
        self._knee_accel_ini = 0.0

    def reset(self, knee_angle: float, knee_vel: float, knee_accel: float,
              t: float) -> None:
        """Take over from phase 2.

        knee_angle [rad], knee_vel [rad/s] and knee_accel [rad/s^2] are phase
        2's last command and its first two rates, t [s] the entry time.
        Returns None.
        """
        self.knee_desired = knee_angle
        self.gamma = 1.0
        self._t_entry = t
        self._t_prev = t - self._dt_nominal
        self._knee_vel_ini = knee_vel
        self._knee_accel_ini = knee_accel

    def compute(self, t: float, hip_angle: float) -> float:
        """Knee angle command [rad] for this tick.

        hip_angle [rad] is the measured hip, clamped below at hip_angle_min.
        """
        cfg = self._cfg
        hip_angle = max(hip_angle, cfg.hip_angle_min)
        dt = t - self._t_prev
        self._t_prev = t

        # both decays are written per baseline tick, so elapsed time over that
        # period gives the same envelope in wall time at any control rate
        n = (t - self._t_entry) / cfg.baseline_dt

        accel = self._knee_accel_ini * np.exp(-cfg.gain_accel * n)
        knee_vel = self._knee_vel_ini + dt * accel

        target = hip_angle - cfg.hip_knee_offset
        vel_target = (target - self.knee_desired) / dt
        self.gamma = np.exp(-cfg.gain_vel * n)
        knee_vel = knee_vel * self.gamma + vel_target * (1.0 - self.gamma)

        self.knee_desired += knee_vel * dt
        return self.knee_desired


class Phase(IntEnum):
    """Which swing phase is driving the knee. IDLE is stance or standing."""
    IDLE = 0
    ONE = 1
    TWO = 2
    THREE = 3


class PhaseChart:
    """Sequences the three phases, from `swing phase chart` in nm_nn_swing.

    Transitions are on leg geometry, not on elapsed time:

        IDLE  -> ONE     swing begins
        ONE   -> TWO     heel swings ahead of the hip, or the prediction turns
                         back toward extension
        TWO   -> THREE   the knee crosses below hip_angle - hip_knee_offset
        any   -> IDLE    standing, or swing ends

    The exit to IDLE outranks the forward transition in every state, matching
    the chart's transition ordering.

        chart.update(...)   once per tick
        chart.phase         what to run, and `entered` if it just changed
    """

    def __init__(self, cfg: NNSwingParams = NNSWING,
                 hip_to_knee: float = LEG.hip_to_knee,
                 heel_radius: float = HEEL_RADIUS,
                 heel_angle: float = HEEL_ANGLE):
        self._cfg = cfg
        self._heel = (hip_to_knee, heel_radius, heel_angle)
        self.phase = Phase.IDLE
        self.entered = False
        self._descending = False

    def reset(self) -> None:
        """Back to IDLE. Returns None."""
        self.phase = Phase.IDLE
        self.entered = False
        self._descending = False

    def update(self, swing: bool, standing: bool, knee_desired: float,
               hip_angle: float, hip_x: float, descending: bool) -> Phase:
        """Advance one tick.

        Args:
            swing, standing: gait and mode flags.
            knee_desired: the current knee command [rad], clamped here the way
                the chart's inputs are.
            hip_angle: filtered hip angle [rad].
            hip_x: hip's horizontal position [m].
            descending: phase 1's flag that its latched trajectory ends below
                where it starts. Used one tick later, since the baseline's
                chart reads it from a data store that phase 1 writes after it.

        Returns:
            the Phase to run this tick. `entered` is true on its first tick.
        """
        cfg = self._cfg
        was = self.phase
        knee = clamp(knee_desired, cfg.knee_min, cfg.knee_max)
        descending, self._descending = self._descending, descending

        if self.phase is Phase.IDLE:
            if swing:
                self.phase = Phase.ONE
        elif standing or not swing:
            self.phase = Phase.IDLE
        elif self.phase is Phase.ONE:
            heel_x = heel_forward(hip_angle, knee, hip_x, *self._heel)
            if heel_x > hip_x or descending:
                self.phase = Phase.TWO
        elif self.phase is Phase.TWO:
            if knee < hip_angle - cfg.hip_knee_offset and hip_angle > 0.0:
                self.phase = Phase.THREE

        if self.phase is Phase.IDLE or (self.phase is Phase.ONE
                                        and was is Phase.IDLE):
            # The baseline zeroes the negativity data store throughout stance,
            # so the chart reads false for the first ticks of every swing.
            # Clearing on entry as well as in IDLE matters because phase 1 is
            # reset by the caller AFTER this runs, so the value handed in on
            # the entry tick is still the previous swing's -- and reading it
            # next tick would hand straight over to phase 2.
            self._descending = False

        self.entered = self.phase is not was
        return self.phase
