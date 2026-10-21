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

    # Sagittal hip and knee positions [m], nav frame: +z up (gravity-aligned),
    # +x the filter's initial heading. Both sit on the thigh link, so the ESKF
    # gets them from the thigh IMU alone -- no knee angle involved. ESKF-owned,
    # same caveat as the hip_* fields: valid only once .initialized.
    # Absolute positions in a frame whose origin and heading are set by wherever
    # the filter started, so only differences over a short window mean anything.
    hip_pos_x: float = 0.0
    hip_pos_z: float = 0.0
    knee_pos_x: float = 0.0
    knee_pos_z: float = 0.0

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
    # Loaded/unloaded, thresholded against a live zero by the proprioceptive
    # estimator. This is a MEASUREMENT, which is why it lives here -- unlike
    # `stance`, which is a control state and stays private to the walking
    # controller. Both the mode FSM and the gait FSM read this one signal.
    contact: bool = False
    # Foot held still by the ESKF's stationarity detector, which is a subset of
    # stance rather than the same thing: it drops at heel-off, well before the
    # foot leaves the ground. ESKF-owned, written every tick once it is
    # running. The NN swing predictor arms on its falling edge, which is what
    # buys a settled prediction before toe-off.
    foot_stationary: bool = False
    grf_R: float = 0.0
    grf_L: float = 0.0

    #derived from kinematics
    leg_angle: float = 0.0
    leg_angle_velocity: float = 0.0
    leg_length: float = 0.0

    # Drives confirmed in CiA-402 OPERATION_ENABLED. A hardware FACT written by
    # the low-level process from ec.control_enabled(); the control process
    # cannot see the EtherCAT master, so this is how it learns torque is
    # actually deliverable. The mode FSM gates INITIALIZING -> STANDING on it,
    # which is the baseline's `[control_enabled && standing_mode_toggle]`.
    control_enabled: bool = False

    # mode fields — supervisor-owned (TBD who sets it), read by high-level
    # selection/edge and by knee conditioning.
    standing: bool = False

