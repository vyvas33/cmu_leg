"""
Uses sensor readings to write the prosthesis's state. 
"""

from cmu_leg.types.state import ProsthesisState
from cmu_leg.types.sensors import RawEtherCATFrame
from cmu_leg.types.sensors import IMUFrame
from cmu_leg.estimation.encoders import convert_encoders
from cmu_leg.control.utils import BackwardDifference

class StateEstimator:
    def __init__(self):
        self._knee_vel = BackwardDifference()
        self._knee_motor_vel = BackwardDifference()
        self._ankle_vel = BackwardDifference()
        self._ankle_motor_vel = BackwardDifference()
        self._ankle_spring_vel = BackwardDifference() ##this needs to be discussed
        self.state = ProsthesisState()


    def get_state(self, frame: RawEtherCATFrame, imu: IMUFrame, t) -> ProsthesisState:
        self._apply_encoders(frame)

        return self.state


    def _apply_encoders(self, frame: RawEtherCATFrame):
        enc_read = convert_encoders(frame=frame)

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

        



        


        