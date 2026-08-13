"""Min-jerk swing: two quintics per joint, tracked by a pid.

Ported from `minjerk swing Control/Min Jerk Swing` in nm_nn_swing. Both
trajectories are solved once, in reset(), from the joint state at toe-off;
compute() only evaluates them.

Per joint, two quintics in swing-relative time:
    segment 1  [0, t_knot]  (theta_0, dtheta_0, ddtheta_0) -> (flex, 0, a_knot)
    segment 2  [t_knot, T]  (flex, 0, a_knot)              -> (extend, 0, 0)

The knee pins a_knot to a cadence-scaled constant, which leaves no constraint
row spanning both segments, so each is a determined 6x6 solve. The ankle
constrains only continuity of a_knot across the knot, leaving one free degree
of freedom for the jerk cost to resolve; that is the coupled solve in
`_solve_coupled`. Both paths live in `_JointTrajectory.solve`.
"""
import numpy as np

import cmu_leg.config.params as params
from cmu_leg.config.params import MJSWING, CommonParams
from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.commands import TorqueDesired
from cmu_leg.control.utils import clamp, PID


def _row_pos(t):
    return [t**5, t**4, t**3, t**2, t, 1.0]


def _row_vel(t):
    return [5*t**4, 4*t**3, 3*t**2, 2*t, 1.0, 0.0]


def _row_acc(t):
    return [20*t**3, 12*t**2, 6*t, 2.0, 0.0, 0.0]


def _h_jerk(ti, tf):
    """Hessian of the jerk cost J = integral of (d3 theta/dt3)^2 over [ti, tf].

    For a quintic only c3, c4, c5 survive three derivatives, so J expands to a
    quadratic in those three alone and the rest of the 6x6 is zero:

        J = 36 c3^2 d1 + 144 c3 c4 d2 + 192 c4^2 d3
            + 240 c3 c5 d3 + 720 c4 c5 d4 + 720 c5^2 d5,     dk = tf^k - ti^k

    The block below is d2J/dc_i dc_j over (c3, c4, c5), placed at descending-power
    indices [2, 1, 0] to match the coefficient order used here. It is rank 3 of
    6, since c0, c1 and c2 do not affect jerk at all, which is why the caller
    adds a ridge before the kkt solve.
    """
    d1, d2, d3, d4, d5 = (tf - ti, tf**2 - ti**2, tf**3 - ti**3,
                          tf**4 - ti**4, tf**5 - ti**5)
    block = np.array([[36*d1,  72*d2, 120*d3],
                      [72*d2, 192*d3, 360*d4],
                      [120*d3, 360*d4, 720*d5]], dtype=float)
    h = np.zeros((6, 6))
    idx = [2, 1, 0]                      # (c3, c4, c5) in descending order
    h[np.ix_(idx, idx)] = block
    return h


class _JointTrajectory:
    """One joint's two-segment swing trajectory.

    Coefficients are descending power, [c5 c4 c3 c2 c1 c0], to match np.polyval.
    The polynomials are in absolute swing time: segment 2 is evaluated at
    t in [t_knot, T] directly, not re-based to zero.
    """

    def __init__(self, flex_angle, extend_angle, flex_frac,
                 angle_min_soft, angle_max_soft):
        self._flex = flex_angle
        self._extend = extend_angle
        self._flex_frac = flex_frac
        self._min_soft = angle_min_soft
        self._max_soft = angle_max_soft
        self.T = None
        self.t_knot = None

    @staticmethod
    def _solve_segment(tstart, tfinal, b):
        """One quintic from six boundary conditions.

        Rows are (pos, vel, accel) at tstart then at tfinal, so `b` must be
        [th0, dth0, ddth0, thf, dthf, ddthf].
        """
        A = np.array([_row_pos(tstart), _row_vel(tstart), _row_acc(tstart),
                      _row_pos(tfinal), _row_vel(tfinal), _row_acc(tfinal)],
                     dtype=float)
        return np.linalg.solve(A, np.array(b, dtype=float))

    @staticmethod
    def _derivatives(c):
        """(angle, vel, accel) coefficient tuples from one angle poly.

        Velocity and acceleration are the exact derivatives of the angle
        polynomial, so they cannot disagree with it. Returned as plain float
        tuples: the solve happens once per swing, these run every tick.
        """
        dc = np.polyder(c)
        ddc = np.polyder(dc)
        return tuple(c.tolist()), tuple(dc.tolist()), tuple(ddc.tolist())

    def _solve_coupled(self, ft, T, angle, velocity, accel):
        """Both segments at once, with the knot acceleration left free.

        11 constraints on 12 unknowns; the leftover degree of freedom is the
        knot acceleration, which the jerk cost picks. Stationarity of a
        quadratic under linear equality constraints is itself linear, so this
        is one 23x23 solve rather than an iteration:

            [H  A^T] [c]   [0]
            [A   0 ] [L] = [b]

        Only the last row spans both segments (acceleration continuity); every
        other row lives in one block.
        """
        z = [0.0] * 6
        aeq = np.array([
            _row_pos(0.0) + z,                                      # theta_0
            _row_pos(ft) + z,                                       # flex
            z + _row_pos(ft),                                       # flex
            z + _row_pos(T),                                        # extend
            _row_vel(0.0) + z,                                      # dtheta_0
            _row_vel(ft) + z,                                       # 0
            z + _row_vel(ft),                                       # 0
            z + _row_vel(T),                                        # 0
            _row_acc(0.0) + z,                                      # ddtheta_0
            z + _row_acc(T),                                        # 0
            [a - b for a, b in zip(_row_acc(ft) + z, z + _row_acc(ft))],
        ], dtype=float)
        beq = np.array([angle, self._flex, self._flex, self._extend,
                        velocity, 0.0, 0.0, 0.0, accel, 0.0, 0.0], dtype=float)

        h = np.zeros((12, 12))
        h[:6, :6] = _h_jerk(0.0, ft)
        h[6:, 6:] = _h_jerk(ft, T)
        # _h_jerk returns half the hessian, so the transpose-add recovers
        # grad^2 J. The ridge covers the c0/c1/c2 null directions so the kkt
        # matrix cannot go singular; 1e-6 against entries of order 1e2-1e3.
        h = h + h.T + np.eye(12) * 1e-6

        n_c, n_e = 12, aeq.shape[0]
        kkt = np.block([[h, aeq.T], [aeq, np.zeros((n_e, n_e))]])
        rhs = np.concatenate([np.zeros(n_c), beq])
        sol = np.linalg.solve(kkt, rhs)
        return sol[:6], sol[6:12]

    def solve(self, angle, velocity, accel, T, knot_accel):
        """Rebuild the trajectory from a toe-off state and a swing duration.

        `knot_accel` None leaves the knot acceleration free and jerk-minimising;
        a number pins it on both sides, which splits the segments into two
        independent solves.
        """
        # Soft-limit guard: outside the limits the angle is clamped and both
        # derivatives are forced to zero, so a toe-off pose out of range cannot
        # inject velocity into the trajectory.
        if angle < self._min_soft or angle > self._max_soft:
            angle = clamp(angle, self._min_soft, self._max_soft)
            velocity = 0.0
            accel = 0.0

        self.T = T
        self.t_knot = self._flex_frac * T

        if knot_accel is None:
            c1, c2 = self._solve_coupled(self.t_knot, T, angle, velocity, accel)
        else:
            c1 = self._solve_segment(0.0, self.t_knot,
                                     [angle, velocity, accel,
                                      self._flex, 0.0, knot_accel])
            c2 = self._solve_segment(self.t_knot, T,
                                     [self._flex, 0.0, knot_accel,
                                      self._extend, 0.0, 0.0])

        self._seg1 = self._derivatives(c1)
        self._seg2 = self._derivatives(c2)

    def setpoint(self, tau):
        """(angle, velocity, accel) at swing-relative time `tau`.

        tau is clamped to [0, T], which is also what holds the joint at the
        extend pose past the end of swing: segment 2 is pinned to (extend, 0, 0)
        at T by its own end constraints, so no separate hold branch is needed.
        """
        tau = clamp(tau, 0.0, self.T)
        c, dc, ddc = self._seg1 if tau < self.t_knot else self._seg2
        return (np.polyval(c, tau), np.polyval(dc, tau), np.polyval(ddc, tau))


class MinJerkSwing:

    _verified = False   # not run on hardware

    def __init__(self, cfg=MJSWING, common: CommonParams = params.COMMON):
        self._cfg = cfg
        self._dt_nominal = 1.0 / common.control_sample_rate
        self.t0 = None
        self._t_prev = None
        self._T = None

        self.knee = _JointTrajectory(cfg.knee_flex_angle, cfg.knee_extend_angle,
                                     cfg.knee_flex_frac,
                                     cfg.knee_angle_min_soft,
                                     cfg.knee_angle_max_soft)
        self.ankle = _JointTrajectory(cfg.ankle_flex_angle,
                                      cfg.ankle_extend_angle,
                                      cfg.ankle_flex_frac,
                                      cfg.ankle_angle_min_soft,
                                      cfg.ankle_angle_max_soft)

        self._knee_pid = PID(cfg.knee_kp, cfg.knee_ki, cfg.knee_kd,
                             cfg.deriv_filter_n)
        self._ankle_pid = PID(cfg.ankle_kp, cfg.ankle_ki, cfg.ankle_kd,
                              cfg.deriv_filter_n)

    def reset(self, state: ProsthesisState, time, prev_stance_duration=None):
        """Latch the toe-off state and solve both trajectories. Called on the
        stance -> swing edge, and on mode entry with no stance duration yet."""
        cfg = self._cfg
        self.t0 = time
        # Seed one nominal period back, not at `time`. The first compute() after
        # a reset lands on the same tick, so `time - _t_prev` would be zero and
        # clamp up to dt_min, giving the pid a half-length first step. That is
        # only visible when the initial error is nonzero, which the soft-limit
        # guard below can produce.
        self._t_prev = time - self._dt_nominal
        # Start each swing from clean pid state, so the integrator does not
        # carry the previous swing's accumulated error into the new one.
        self._knee_pid.reset()
        self._ankle_pid.reset()

        # Swing duration tracks cadence off the stance that just ended, and is
        # shared by both joints; only the knot fraction differs.
        #
        # On mode entry no stance has completed, so there is no cadence to
        # track and a duration has to be picked. Take the slowest the clamp
        # allows: this swing runs the instant the operator asks for walking,
        # from a pose the soft-limit guard may well have clamped, so it is the
        # one that should be gentlest. The baseline instead lands on the clamp
        # floor here, because both its latch times are still zero.
        if prev_stance_duration is None:
            self._T = cfg.swing_duration_max
        else:
            self._T = clamp(cfg.swing_to_stance_ratio * prev_stance_duration,
                            cfg.swing_duration_min, cfg.swing_duration_max)

        # Knee knot acceleration: the Winter peak-flexion value, time-scaled to
        # this swing duration by the 1/t^2 law.
        knee_a_knot = (cfg.winter_flex_knee_accel
                       / (self._T / cfg.winter_swing_duration) ** 2)

        self.knee.solve(state.knee_angle, state.knee_velocity, state.knee_accel,
                        self._T, knee_a_knot)
        # None: leave the ankle knot acceleration to the jerk cost.
        self.ankle.solve(state.ankle_angle, state.ankle_velocity,
                         state.ankle_accel, self._T, None)

    def compute(self, state, time):
        tau = time - self.t0
        # The pid takes the position error. The commanded velocity still reaches
        # its d term, since the pid differentiates that error rather than the
        # measurement.
        knee_d, _, _ = self.knee.setpoint(tau)
        ankle_d, _, _ = self.ankle.setpoint(tau)

        dt = time - self._t_prev
        self._t_prev = time

        return TorqueDesired(
            knee_desired=self._knee_pid.compute(knee_d - state.knee_angle, dt),
            ankle_desired=self._ankle_pid.compute(ankle_d - state.ankle_angle, dt),
        )
