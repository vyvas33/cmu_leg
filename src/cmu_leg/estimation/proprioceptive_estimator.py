"""
Proprioceptive state estimation.

Converts one EtherCAT frame into the prosthesis's proprioceptive physical state
-- joint and motor angles/velocities, torques, spring, and the hall GRF proxies
-- using the leg's own body sensors (as opposed to the IMUs) and their software
derivatives (backward-difference velocities). Those sensors span two boards: the
medulla MCU (joint encoders, hall sensors) and the Elmo drives (motor encoders,
motor torque), so this is broader than any one board.

This is the half of the estimator that runs in the low-level process; the ESKF
(IMU-derived hip_*) half runs in the estimator process. ProsthesisStateEstimator composes
this with the ESKF for the single-process path. The low-level process publishes
this full proprioceptive state to the estimator over B1 (a StateBuffer); the
estimator overlays hip_* and republishes the complete state on B2.
"""

from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.sensors import RawEtherCATFrame
from cmu_leg.estimation.encoders import convert_encoders
from cmu_leg.control.utils import BackwardDifference
from cmu_leg.control.utils import MedianFilter
from cmu_leg.control.utils import Lowpass
from cmu_leg.config.params import ENCODERS, COMMON


class ProprioceptiveEstimator:
    """Sensor -> proprioceptive state. Owns the per-tick filters (velocity
    backward differences, ankle median, GRF lowpass) and the ProsthesisState bus
    it writes into."""

    def __init__(self):
        self._knee_vel = BackwardDifference()
        self._knee_motor_vel = BackwardDifference()
        self._ankle_vel = BackwardDifference()
        self._ankle_motor_vel = BackwardDifference()
        self._ankle_spring_vel = BackwardDifference()  ##this needs to be discussed
        self._ankle_median = MedianFilter()

        # hall GRF proxy lowpass (baseline: Cheby-II 10 Hz; Butterworth stand-in)
        self._toe_grf_lp = Lowpass(cutoff_freq=10.0, sample_rate=COMMON.sample_rate, order=2)
        self._heel_grf_lp = Lowpass(cutoff_freq=10.0, sample_rate=COMMON.sample_rate, order=2)

        self.state = ProsthesisState()

    def update(self, frame: RawEtherCATFrame, dt: float) -> ProsthesisState:
        """Write the proprioceptive fields for one tick and return the bus.

        Args:
            frame (RawEtherCATFrame): Raw PDO inputs for this cycle.
            dt (float): Measured elapsed time since the previous tick (s), used
                by every backward-difference velocity.

        Returns:
            state (ProsthesisState): The same bus instance, fields updated.
        """
        self._apply_encoders(frame, dt)
        self._apply_hall_grfs(frame)
        return self.state

    def _apply_hall_grfs(self, frame: RawEtherCATFrame):
        toe = ENCODERS.hall_0_offset - frame.hall_0
        heel = ENCODERS.hall_1_offset - frame.hall_1

        self.state.toe_grf = self._toe_grf_lp.compute(toe)
        self.state.heel_grf = self._heel_grf_lp.compute(heel)

    def _apply_encoders(self, frame: RawEtherCATFrame, dt: float):
        enc_read = convert_encoders(frame=frame, ankle_median=self._ankle_median)

        knee_angle = enc_read.knee_angle
        knee_motor_angle = enc_read.knee_motor_angle
        ankle_angle = enc_read.ankle_angle
        ankle_motor_angle = enc_read.ankle_motor_angle
        knee_torque = enc_read.knee_torque
        ankle_torque = enc_read.ankle_torque

        ankle_spring_angle = enc_read.ankle_spring_angle

        knee_velocity = self._knee_vel.compute(knee_angle, dt)
        knee_motor_velocity = self._knee_motor_vel.compute(knee_motor_angle, dt)

        ankle_velocity = self._ankle_vel.compute(ankle_angle, dt)
        ankle_motor_velocity = self._ankle_motor_vel.compute(ankle_motor_angle, dt)
        ankle_spring_velocity = self._ankle_spring_vel.compute(ankle_spring_angle, dt)

        self.state.knee_angle = knee_angle
        self.state.knee_velocity = knee_velocity
        self.state.ankle_angle = ankle_angle
        self.state.ankle_velocity = ankle_velocity
        self.state.ankle_spring_angle = ankle_spring_angle
        self.state.ankle_spring_velocity = ankle_spring_velocity

        self.state.knee_torque_measured = knee_torque
        self.state.ankle_torque_measured = ankle_torque

        self.state.knee_motor_angle = knee_motor_angle
        self.state.knee_motor_velocity = knee_motor_velocity
        self.state.ankle_motor_angle = ankle_motor_angle
        self.state.ankle_motor_velocity = ankle_motor_velocity
