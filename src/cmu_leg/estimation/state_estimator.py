"""
Uses sensor readings to write the prosthesis's state. 
"""

from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.sensors import RawEtherCATFrame
from cmu_leg.types.sensors import IMUFrame
from cmu_leg.estimation.encoders import convert_encoders
from cmu_leg.control.utils import BackwardDifference
from cmu_leg.control.utils import MedianFilter
from cmu_leg.control.utils import Lowpass
from cmu_leg.config.params import ENCODERS, COMMON

class StateEstimator:
    def __init__(self):
        self._knee_vel = BackwardDifference()
        self._knee_motor_vel = BackwardDifference()
        self._ankle_vel = BackwardDifference()
        self._ankle_motor_vel = BackwardDifference()
        self._ankle_spring_vel = BackwardDifference() ##this needs to be discussed
        self.state = ProsthesisState()
        self._ankle_median = MedianFilter()

        # hall GRF proxy lowpass (baseline: Cheby-II 10 Hz; Butterworth stand-in)
        self._toe_grf_lp = Lowpass(cutoff_freq=10.0, sample_rate=COMMON.sample_rate, order=2)
        self._heel_grf_lp = Lowpass(cutoff_freq=10.0, sample_rate=COMMON.sample_rate, order=2)


    def get_state(self, frame: RawEtherCATFrame, imu: IMUFrame, t) -> ProsthesisState:
        self._apply_encoders(frame)
        self._apply_hall_grfs(frame)

        return self.state

    def _apply_hall_grfs(self, frame: RawEtherCATFrame):
        toe = ENCODERS.hall_0_offset - frame.hall_0
        heel = ENCODERS.hall_1_offset - frame.hall_1

        self.state.toe_grf = self._toe_grf_lp.compute(toe)
        self.state.heel_grf = self._heel_grf_lp.compute(heel)

    def _apply_encoders(self, frame: RawEtherCATFrame):
        enc_read = convert_encoders(frame=frame, ankle_median=self._ankle_median)

        knee_angle = enc_read.knee_angle
        knee_motor_angle = enc_read.knee_motor_angle
        ankle_angle = enc_read.ankle_angle
        ankle_motor_angle = enc_read.ankle_motor_angle
        knee_torque = enc_read.knee_torque
        ankle_torque = enc_read.ankle_torque

        ankle_spring_angle = enc_read.ankle_spring_angle

        knee_velocity = self._knee_vel.compute(knee_angle)
        knee_motor_velocity = self._knee_motor_vel.compute(knee_motor_angle)

        ankle_velocity = self._ankle_vel.compute(ankle_angle)
        ankle_motor_velocity= self._ankle_motor_vel.compute(ankle_motor_angle)
        ankle_spring_velocity = self._ankle_spring_vel.compute(ankle_spring_angle)

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

        



        


        