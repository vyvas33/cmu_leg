from scipy.signal import butter, sosfilt, sosfilt_zi
from cmu_leg.config.params import COMMON
import collections
import statistics


def clamp(x: float, lo: float, up: float) -> float:
    """Scalar saturation. Same result as np.clip(x, lo, up) in all orderings
    (np.clip is min(max(x, lo), up) internally), but keeps native floats and
    allocates nothing -- np.clip on a scalar returns a numpy float64 that then
    slows every downstream arithmetic op. Cheap enough for the 1 kHz loop."""
    return min(max(x, lo), up)

class Lowpass:
    """Butterworth lowpass as a scalar biquad cascade (transposed direct-form II).

    Coefficients come from scipy.butter (design-time only). The per-sample filter
    is hand-rolled pure-Python scalar math instead of calling scipy.sosfilt on a
    one-element array every tick, which is ~15x cheaper per sample and allocates
    nothing. Output is numerically identical to sosfilt with the same `zi` state.
    """
    def __init__(self,
                 cutoff_freq: float = 25.0,
                 sample_rate: float = 1000.0,
                 order: int = 2):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        sos = butter(order, normal_cutoff, btype='low', output='sos')
        zi_unit = sosfilt_zi(sos)               # steady state for a unit-step input

        # per-section coefficients (a0 is normalized to 1 in SOS form) and the
        # unit-step steady-state initial conditions, as plain Python floats.
        self._coeffs = [(float(b0), float(b1), float(b2), float(a1), float(a2))
                        for b0, b1, b2, _, a1, a2 in sos]
        self._zi_unit = [(float(s0), float(s1)) for s0, s1 in zi_unit]

        # filter state per section: z1, z2 (transposed DF2 delay registers)
        self._z1 = [0.0] * len(self._coeffs)    # start from zero
        self._z2 = [0.0] * len(self._coeffs)

    # call reset whenever control_enabled changes to true.
    # Seeds the filter state so that, for a constant input u0, the filter
    # output is exactly u0 with no startup transient.
    def reset(self, u0: float = 0.0) -> None:
        self._z1 = [s0 * u0 for s0, _ in self._zi_unit]
        self._z2 = [s1 * u0 for _, s1 in self._zi_unit]

    def compute(self, u: float) -> float:
        x = u
        z1 = self._z1
        z2 = self._z2
        for k, (b0, b1, b2, a1, a2) in enumerate(self._coeffs):
            y = b0 * x + z1[k]
            z1[k] = b1 * x - a1 * y + z2[k]
            z2[k] = b2 * x - a2 * y
            x = y
        return x

class BackwardDifference:
    """Discrete derivative on the caller's MEASURED dt.

    dt is required, not defaulted to COMMON.sample_time, on purpose: a missed
    call site should be a TypeError, not a silently wrong velocity. The loop
    does overrun -- a logged 21 ms tick divided by a nominal 1 ms would report
    21x the true rate, straight into the controllers.

    dt is clamped to [COMMON.dt_min, COMMON.dt_max]; see CommonParams for why
    both ends matter.
    """

    def __init__(self, dt_min: float = COMMON.dt_min, dt_max: float = COMMON.dt_max):
        self._dt_min = dt_min
        self._dt_max = dt_max
        self._u_prev: float | None = None

    # call reset whenever control_enabled changes to true
    def reset(self, u0: float = 0.0) -> None:
        # seed the delay with u0 so the next compute() returns 0
        self._u_prev = u0

    def compute(self, u: float, dt: float) -> float:
        if self._u_prev is None:
            self._u_prev = u
        dt = min(max(dt, self._dt_min), self._dt_max)
        y = (u - self._u_prev) / dt
        self._u_prev = u
        return y

class PID:
    """Discrete pid: forward Euler on both the integrator and the derivative
    filter, filter coefficient n, no output saturation and no anti-windup.

        u   = kp*e + xi + n*(kd*e - xf)      <- pre-update states
        xi += dt * ki * e
        xf += dt * n*(kd*e - xf)

    Taking the output from the pre-update states is what makes it forward Euler;
    updating first would be backward Euler and shift the effective gains by a
    tick.

    The filter recursion realises kd*s/(1 + s/n), so xf tracks the derivative
    through a first-order lag and n is the derivative bandwidth in rad/s.

    `error` is setpoint minus measurement, so the d term acts on the error and
    responds to a moving setpoint. Damping the measurement instead brakes
    commanded motion; see StandingController for a case where that is wanted.

    dt is required and clamped to [COMMON.dt_min, COMMON.dt_max], same reasoning
    as BackwardDifference: a stalled tick must not integrate as one big step.
    """

    def __init__(self, kp: float, ki: float, kd: float, n: float,
                 dt_min: float = COMMON.dt_min, dt_max: float = COMMON.dt_max):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._n = n
        self._dt_min = dt_min
        self._dt_max = dt_max
        self._xi = 0.0
        self._xf = 0.0

    def reset(self) -> None:
        """Zero the integrator and filter states. Call on entry into the phase
        this pid serves, so nothing carries across from the previous one."""
        self._xi = 0.0
        self._xf = 0.0

    def compute(self, error: float, dt: float) -> float:
        dt = min(max(dt, self._dt_min), self._dt_max)
        d_out = self._n * (self._kd * error - self._xf)
        u = self._kp * error + self._xi + d_out
        self._xi += dt * self._ki * error
        self._xf += dt * d_out
        return u


class MedianFilter:
    def __init__(self, window: int = 3):
        self._buffer = collections.deque(maxlen=window)

    def reset(self):
        self._buffer.clear()

    def compute(self, x):
        self._buffer.append(x)
        return statistics.median(self._buffer)


