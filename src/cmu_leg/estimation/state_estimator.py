"""
Uses sensor readings to write the prosthesis's state.

Composes the two halves of estimation for the single-process path:
  - ProprioceptiveEstimator: proprioceptive physical state (encoders, halls,
    backward-difference velocities). Runs in the low-level process.
  - ESKF: IMU-derived hip_* via error-state Kalman filter. Runs in the
    estimator process.
Both write into ONE ProsthesisState bus (self.state is the proprioceptive
estimator's state instance; the ESKF writes hip_* into that same object).
get_state() runs them in order and stays the single-process fallback +
byte-identical verification reference; the two runtime processes each drive one
half directly.
"""

from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.sensors import RawEtherCATFrame
from cmu_leg.types.sensors import IMUFrame
from cmu_leg.estimation.proprioceptive_estimator import ProprioceptiveEstimator
from cmu_leg.control.utils import BackwardDifference
from cmu_leg.estimation.eskf.eskf import ESKF


class ProsthesisStateEstimator:
    def __init__(self):
        self._proprio = ProprioceptiveEstimator()
        self._eskf = ESKF()
        self._hip_accel = BackwardDifference()
        # Share the proprioceptive estimator's bus: the ESKF writes hip_* into
        # this same instance. ProprioceptiveEstimator only mutates fields in
        # place, never rebinds .state, so the alias holds for the estimator's life.
        self.state = self._proprio.state

    def get_state(self, frame: RawEtherCATFrame, imu: IMUFrame, t: float,
                  dt: float) -> ProsthesisState:
        """t is wall-clock (perf_counter) seconds, sharing the IMU driver's
        timebase -- the ESKF's staleness check compares them directly. dt is
        the MEASURED elapsed time since the previous tick, not the nominal
        period; everything that differentiates or integrates uses it.
        """
        self._proprio.update(frame, dt)
        self._apply_eskf(imu, t, dt)

        return self.state

    def _apply_eskf(self, imu: IMUFrame, t: float, dt: float):
        # update() writes hip_angle / hip_velocity onto self.state itself, and
        # holds the previous values until its init window closes.
        self._eskf.update(imu, self.state, t, dt)
        if self._eskf.initialized:
            self.state.hip_accel = self._hip_accel.compute(self.state.hip_velocity, dt)
