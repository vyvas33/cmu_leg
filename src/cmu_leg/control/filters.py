from scipy.signal import butter, sosfilt, sosfilt_zi

class Lowpass:
    def __init__(self, 
                 cutoff_freq: float = 25.0, 
                 sample_rate: float = 1000.0, 
                 order: int = 2):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        self._sos = butter(order, normal_cutoff, 
                          btype='low', output='sos')
        self._zi = sosfilt_zi(self._sos)
    
    def compute(self, u: float) -> float:
        y, self._zi = sosfilt(self._sos, [u], zi=self._zi)
        return float(y[0])
