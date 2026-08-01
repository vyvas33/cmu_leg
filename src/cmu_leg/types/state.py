from dataclasses import dataclass

@dataclass
class ProsthesisState:
    """The control bus. Two writers with disjoint field ownership:
      - physical fields (angles/velocities/torques/grf/kinematics) — written by
        the ProsthesisStateEstimator from sensors. The hip_* fields are the exception in
        practice: the ESKF that ProsthesisStateEstimator owns writes them itself, from
        the thigh IMU, and only once its init window has closed (see
        ESKF.initialized) — before that they hold their previous value.
      - mode fields (below) — written by the supervisor, NOT the estimator.
    `stance`/swing is deliberately NOT here: it lives inside the walking
    controller and never leaves it.
    """
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

    # mode fields — supervisor-owned (TBD who sets it), read by high-level
    # selection/edge and by knee conditioning.
    standing: bool = False

