from dataclasses import dataclass


@dataclass(slots=True)
class RawEtherCATFrame:
    """Raw integer PDO inputs for one EtherCAT cycle, as they arrive on the bus. 
    See hardware/README.md for the byte/type/source map.
    Defaults let a replay harness build partial frames.
    """

    # Knee Elmo
    knee_motor_counts: int = 0
    knee_digital_inputs: int = 0
    knee_status_word: int = 0
    knee_motor_torque_raw: int = 0
    knee_analog: int = 0

    # Ankle Elmo
    ankle_motor_counts: int = 0
    ankle_digital_inputs: int = 0
    ankle_status_word: int = 0
    ankle_motor_torque_raw: int = 0
    ankle_analog: int = 0

    # Medulla
    medulla_id: int = 0
    medulla_state: int = 0
    medulla_counter: int = 0
    medulla_error_flags: int = 0
    medulla_limit_switch: int = 0
    knee_joint_counts: int = 0
    knee_joint_timestamp: int = 0
    ankle_joint_counts: int = 0
    ankle_joint_timestamp: int = 0
    hall_0: int = 0
    hall_1: int = 0
    medulla_logic_voltage: int = 0

@dataclass(slots=True)
class IMUFrame:
    """SI-unit IMU outputs in each sensor's OWN frame (no EKF remap). 
    Per-sensor pll timestamp + freshness flag.
    """
    thigh_acc: tuple[float, float, float] = (0.0, 0.0, 0.0)   # m/s²
    thigh_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s
    foot_acc: tuple[float, float, float] = (0.0, 0.0, 0.0)    # m/s²
    foot_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)   # rad/s
    thigh_time: float = 0.0    # host_time_s from the driver
    foot_time: float = 0.0
    thigh_fresh: bool = False  # True if newer than the previous read()
    foot_fresh: bool = False

@dataclass(slots=True)
class EncoderReadings:
    """Engineering-unit outputs of the encoder conversion (radians, Nm).
    Pure function of one RawEtherCATFrame; no temporal state (velocities are
    derived downstream in the estimator). See estimation/encoders.py.
    """
    knee_motor_angle: float = 0.0
    knee_angle: float = 0.0
    knee_torque: float = 0.0
    ankle_motor_angle: float = 0.0
    ankle_spring_angle: float = 0.0
    ankle_angle: float = 0.0
    ankle_torque: float = 0.0

@dataclass(slots=True)
class TreadmillFrame:
    """SI-unit belt state from the Bertec Control Panel's status packet.
    Accelerations are not reported. Freshness flag as in IMUFrame.
    """
    right_speed: float = 0.0   # m/s
    left_speed: float = 0.0    # m/s
    incline: float = 0.0       # deg
    time: float = 0.0          # host time.monotonic() the packet was read
    fresh: bool = False        # True if a new packet arrived since the previous read()
