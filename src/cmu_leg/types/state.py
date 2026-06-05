from dataclasses import dataclass

@dataclass
class ProsthesisState:
    hip_angle: float = 0.0
    hip_velocity: float = 0.0
    hip_accel: float = 0.0

    #knee motor 
    knee_motor_angle: float = 0.0
    knee_motor_velocity: float = 0.0
    knee_motor_accel: float = 0.0
    knee_motor_torque_measured: float = 0.0

    #knee joint 
    knee_angle: float = 0.0
    knee_velocity: float = 0.0
    knee_accel: float = 0.0
    knee_torque_measured: float = 0.0

    #ankle 
    ankle_motor_angle: float = 0.0
    ankle_motor_velocity: float = 0.0
    ankle_motor_accel: float = 0.0
    ankle_motor_torque_measured: float = 0.0

    #ankle spring (SEA)
    ankle_spring_angle: float = 0.0
    ankle_spring_velocity: float = 0.0
    ankle_spring_accel: float = 0.0

    ankle_angle: float = 0.0
    ankle_velocity: float = 0.0
    ankle_accel: float = 0.0
    ankle_torque_measured: float = 0.0

    #grf
    toe_grf: float = 0.0
    heel_grf: float = 0.0
    grf_R: float = 0.0
    grf_L: float = 0.0

    #derived from kinematics 
    leg_angle: float = 0.0
    leg_angle_velocity: float = 0.0
    leg_length: float = 0.0

    stance: bool = False
    standing: bool = False
    zero_torque: bool = False
    control_enabled: bool = False
    initializing: bool = False

    vicon_sync: bool #need to confirm this




