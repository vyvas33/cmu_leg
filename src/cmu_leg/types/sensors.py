from dataclasses import dataclass


@dataclass(slots=True)
class RawEtherCATFrame:
    """Raw integer PDO inputs for one EtherCAT cycle, as they arrive on the bus
    (before any scaling). See hardware/README.md for the byte/type/source map.
    Defaults let a replay harness build partial frames."""

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
