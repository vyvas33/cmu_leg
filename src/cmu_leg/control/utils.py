from scipy.signal import butter, sosfilt, sosfilt_zi
from cmu_leg.config.params import COMMON

class Lowpass:
    def __init__(self,
                 cutoff_freq: float = 25.0,
                 sample_rate: float = 1000.0,
                 order: int = 2):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        self._sos = butter(order, normal_cutoff,
                          btype='low', output='sos')
        self._zi_unit = sosfilt_zi(self._sos)   # steady-state for unit step
        self._zi = self._zi_unit * 0.0          # start from zero

    # call reset whenever control_enabled changes to true.
    # Seeds the filter state so that, for a constant input u0, the filter
    # output is exactly u0 with no startup transient.
    def reset(self, u0: float = 0.0) -> None:
        self._zi = self._zi_unit * u0

    def compute(self, u: float) -> float:
        y, self._zi = sosfilt(self._sos, [u], zi=self._zi)
        return float(y[0])

class BackwardDifference:
    def __init__(self, sample_time: float = COMMON.sample_time):
        self._Ts = sample_time
        self._u_prev: float | None = None

    # call reset whenever control_enabled changes to true
    def reset(self, u0: float = 0.0) -> None:
        # seed the delay with u0 so the next compute() returns 0
        self._u_prev = u0

    def compute(self, u: float) -> float:
        if self._u_prev is None:
            self._u_prev = u
        y = (u - self._u_prev) / self._Ts
        self._u_prev = u
        return y
