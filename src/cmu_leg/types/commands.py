from dataclasses import dataclass

#the output of the low level controller 
@dataclass
class LowLevelOutput:
    # knee
    knee_torque_commanded: float = 0.0
    knee_velocity_desired: float = 0.0
    knee_velocity_commanded: float = 0.0
    knee_motor_torque_ff: float = 0.0
    # ankle
    ankle_torque_commanded: float = 0.0
    ankle_velocity_desired: float = 0.0
    ankle_velocity_commanded: float = 0.0
    ankle_motor_torque_ff: float = 0.0

#the command that goes to the motors 
@dataclass
class DriveCommand:
    # knee 
    knee_velocity: float = 0.0
    knee_torque_ff: float = 0.0
    # ankle
    ankle_velocity: float = 0.0
    ankle_torque_ff: float = 0.0

#the output of the high level controller
@dataclass
class TorqueDesired:
    knee_desired: float = 0.0
    ankle_desired: float = 0.0
