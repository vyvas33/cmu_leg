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

#the command that goes to the motors. Hardware-ready values in the units the
#Elmo PDOs natively consume:
#  velocity_counts:   encoder counts per second (Target Velocity, 0x60FF, int32)
#  torque_per_mille:  per-mille of rated torque (Target Torque,   0x6071, int16)
@dataclass
class DriveCommand:
    # knee
    knee_velocity_counts: int = 0
    knee_torque_per_mille: int = 0
    # ankle
    ankle_velocity_counts: int = 0
    ankle_torque_per_mille: int = 0

#the output of the high level controller
@dataclass
class TorqueDesired:
    knee_desired: float = 0.0
    ankle_desired: float = 0.0
