"""
Neural network predictor for swing phase.

Ported from the `NN` subsystem in nm_nn_swing. Predicts the knee's next 100 ms
from the last 100 ms of leg motion, and knows nothing about torque: the swing
controller latches a trajectory out of here and tracks it with a pid.

    update(state, knee_desired, t)     every tick, stance and swing
    predict()                          only at a latch (~20 Hz)

update() just fills the history buffer, so it has to keep running through
stance -- the network needs 100 ms of warm history at toe-off, and the baseline
starts it a few hundred ms before that. NMNPC owns those calls.

The network works in human joint space at both ends: it was trained on
able-bodied gait, so the knee angle going in is remapped to a human leg by
_leg_to_human, and the prediction comes back through _human_to_leg.

The knee channel is the commanded knee, not the measured one: in stance that
is the measured angle (NMNPC passes it), in swing it is the controller's own
output from the previous tick. So during swing the predictor runs on its own
past output for that one channel, while the hip and position channels stay
real.
"""
from pathlib import Path

import numpy as np

from cmu_leg.config.params import COMMON, CommonParams
from cmu_leg.structures.state import ProsthesisState
from cmu_leg.control.high_level.walking import retargeting

# Weights and the constants they were trained with. Built by
# tools/build_nn_weights.py; changing any of these requires retraining, not
# tuning.
_W = np.load(Path(__file__).resolve().parents[3]
             / "config/data/nn_swing_weights.npz")

# 126 -> 128 -> 64 -> 64 -> 41, relu between layers, linear out.
# Biases are flattened to (n,) so the forward pass stays 1-D.
LAYERS = tuple((_W[f"A{i}"], _W[f"b{i}"].ravel()) for i in (1, 2, 3, 4))

MEAN, STD = _W["mean"], _W["std"]              # (6,) each, in CHANNELS order
KNEE_MEAN, KNEE_STD = float(_W["knee_mean"]), float(_W["knee_std"])

# (100, 41): resamples the prediction onto 1 ms and keeps the future half, in
# one matmul. Replaces MATLAB's spline() + 102:201 slice.
SPLINE_FUTURE = _W["spline_future"]

CHANNELS = tuple(str(c) for c in _W["channels"])
# update() writes the row by index, so a reordered .npz has to fail loudly
# rather than feed the network shuffled channels.
if CHANNELS != ("hip_angle", "knee_angle", "hip_x", "hip_z", "knee_x", "knee_z"):
    raise ValueError(f"unexpected channel order in the weights file: {CHANNELS}")
N_CH = len(CHANNELS)                           # 6
_Z_COLS = [CHANNELS.index("hip_z"), CHANNELS.index("knee_z")]
N_HIST = int(_W["n_hist"])                     # 21 snapshot taps
HIST_DT = float(_W["hist_dt"])                 # 5 ms apart
HISTORY_SPAN = (N_HIST - 1) * HIST_DT          # 100 ms of history per snapshot
PLAYBACK_DT = float(_W["playback_dt"])         # 1 ms, spacing of predict()'s output

# Positions are fed to the network in mm; our bus carries metres.
POS_SCALE = 1000.0
# Hip and knee heights are divided by this before standardizing -- subject-height
# normalization from the training set.
HIP_HEIGHT_RATIO = 1.4

KNEE_ANGLE_MAX_SOFT = np.deg2rad(85.0)         # prediction clamp, leg frame

# extra rows so a slow tick cannot overwrite the oldest sample a snapshot needs
_BUFFER_MARGIN = 4
# A gap longer than this between consecutive samples throws the history away.
# _snapshot interpolates between rows, so a hole would be filled with a
# straight line rather than reported. A few missed ticks interpolate fine; a
# scheduler stall does not.
MAX_SAMPLE_GAP = 0.010      # s


class NeuralPredictor:
    def __init__(self, common: CommonParams = COMMON):
        """Allocate the history ring: one row per control tick, covering
        HISTORY_SPAN seconds (55 rows at 500 Hz, 105 at 1 kHz). The extra
        _BUFFER_MARGIN rows keep a few slow ticks from dropping samples that a
        snapshot still needs."""
        n = int(np.ceil(HISTORY_SPAN * common.control_sample_rate)) + 1
        self._cap = n + _BUFFER_MARGIN
        self._buf = np.zeros((self._cap, N_CH))   # channels in CHANNELS order
        self._t = np.zeros(self._cap)             # timestamp of each row
        self._i = 0          # next row to write
        self._count = 0      # rows written so far, saturating at _cap

    @property
    def ready(self) -> bool:
        """Whether predict() can be called yet.

        Returns:
            bool, true once the history covers HISTORY_SPAN.

        Measured from the timestamps, not the row count, so a run of short
        ticks cannot fill the ring without actually reaching back far enough.
        Gaps are handled in update(), which resets rather than leaving one.
        """
        if self._count < 2:
            return False
        oldest = 0 if self._count < self._cap else self._i
        newest = (self._i - 1) % self._cap
        return self._t[newest] - self._t[oldest] >= HISTORY_SPAN

    def reset(self) -> None:
        """Discard the history.

        Takes no arguments, returns None. `ready` stays false until
        HISTORY_SPAN of new samples has been pushed. The old rows are left in
        place; the counters make them unreachable.
        """
        self._i = 0
        self._count = 0

    def update(self, state: ProsthesisState, knee_desired: float,
               t: float) -> None:
        """Append one sample of the six input channels to the history.

        Args:
            state: the control bus. Read for hip_angle [rad] and the hip
                position [m]; the knee position is derived from those, not
                taken from the bus. Positions are scaled to mm on the way in.
            knee_desired: commanded knee angle [rad] in our leg's frame, stored
                remapped to the human leg. The measured angle while in stance,
                the swing controller's previous output once swinging.
            t: time of this sample [s]. Must not go backwards.

        Returns None.

        Call every control tick, in stance and in swing: the network needs a
        full HISTORY_SPAN of history before the first latch of a swing. A gap
        of more than MAX_SAMPLE_GAP since the previous sample discards the
        history rather than leaving a hole for _snapshot to interpolate across.
        """
        if self._count and t - self._t[(self._i - 1) % self._cap] > MAX_SAMPLE_GAP:
            self.reset()

        row = self._buf[self._i]                  # write in place, no temporary
        row[0] = state.hip_angle
        row[1] = self._leg_to_human(knee_desired)
        row[2] = state.hip_pos_x * POS_SCALE
        row[3] = state.hip_pos_z * POS_SCALE
        # The knee position is NOT the estimator's. The network is fed a knee
        # hanging off the hip on the virtual leg's thigh, at the hip angle --
        # `data transfer` in the baseline's Get Prosthesis State. Using the
        # real knee instead costs 5x the prediction error, because the link is
        # 0.46 m here against a real thigh of 0.71.
        row[4] = (state.hip_pos_x
                  + retargeting.L_HUMAN * np.sin(state.hip_angle)) * POS_SCALE
        row[5] = (state.hip_pos_z
                  - retargeting.L_HUMAN * np.cos(state.hip_angle)) * POS_SCALE
        self._t[self._i] = t

        self._i = (self._i + 1) % self._cap
        self._count = min(self._count + 1, self._cap)

    def predict(self) -> np.ndarray:
        """Predict the knee's next 100 ms from the history.

        Takes no arguments; everything comes from the buffer update() fills.

        Returns:
            (100,) knee angles [rad] in our leg's frame, PLAYBACK_DT apart,
            the first one a step after the newest sample. Absolute angles as
            the network produced them -- the caller re-anchors them onto its
            own current command.

        Only valid once `ready`. Positions are fed in relative to the newest
        sample, so the network sees how far the leg travelled over the window
        rather than where it is, and heights are divided by HIP_HEIGHT_RATIO,
        the training set's subject-height normalization.

        The frame change back to our leg happens before the resample, matching
        the baseline: it is a nonlinear map, so transforming the 41 points and
        then splining is not the same as splining and then transforming.
        """
        knee_human = self._forward(self._inputs())
        return SPLINE_FUTURE @ self._human_to_leg(knee_human)

    def _inputs(self) -> np.ndarray:
        """Normalized network input, (N_HIST*N_CH,), timestep-major."""
        # _snapshot returns a fresh array, so the steps below can be in-place
        # without touching the buffer
        x = self._snapshot()
        x[:, 2:] -= x[-1, 2:]                       # positions relative to now
        x[:, _Z_COLS] /= HIP_HEIGHT_RATIO           # heights only
        x -= MEAN
        x /= STD
        return x.reshape(-1)

    def _snapshot(self) -> np.ndarray:
        """Resample the history onto the network's input grid.

        Returns (N_HIST, N_CH) in raw units, oldest row first, last row at the
        newest sample's time. Taps are HIST_DT apart and are interpolated from
        the row timestamps, so the control period need not divide HIST_DT.

        Only valid once `ready`: np.interp clamps at the ends, so a buffer
        holding less than HISTORY_SPAN returns the oldest sample repeated.
        """
        if self._count < self._cap:
            # ring has not wrapped yet, so rows 0.._count are already in order
            t_ord, buf_ord = self._t[:self._count], self._buf[:self._count]
        else:
            # oldest row is the one about to be overwritten
            order = (self._i + np.arange(self._cap)) % self._cap
            t_ord, buf_ord = self._t[order], self._buf[order]

        targets = t_ord[-1] - HISTORY_SPAN + np.arange(N_HIST) * HIST_DT

        out = np.empty((N_HIST, N_CH))
        for c in range(N_CH):
            out[:, c] = np.interp(targets, t_ord, buf_ord[:, c])
        return out

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Run the MLP on one normalized input vector.

        x is (N_HIST*N_CH,), the snapshot flattened timestep-major. Returns
        (41,) knee angles in the human frame, radians, de-standardized.
        """
        h = x
        for a, b in LAYERS[:-1]:
            h = np.maximum(a @ h + b, 0.0)
        a, b = LAYERS[-1]
        return (a @ h + b) * KNEE_STD + KNEE_MEAN

    @staticmethod
    def _leg_to_human(knee_leg: float) -> float:
        """Our knee angle -> the virtual human leg's, radians."""
        return retargeting.knee_leg_to_human(knee_leg)

    @staticmethod
    def _human_to_leg(knee_human):
        """The network's predicted angles -> ours, clamped.

        Takes the whole (41,) prediction at once.
        """
        return retargeting.knee_human_to_leg(knee_human, KNEE_ANGLE_MAX_SOFT)
