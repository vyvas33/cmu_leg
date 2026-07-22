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

class MedianFilter:
    def __init__(self, window: int = 3):
        self._buffer = collections.deque(maxlen=window)

    def reset(self):
        self._buffer.clear()

    def compute(self, x):
        self._buffer.append(x)
        return statistics.median(self._buffer)
        