"""Python port of the Simulink ``read_encoders`` model.

Turns raw EtherCAT PDO integers (``RawSensorFrame``) into engineering-unit
sensor readings (``SensorReadings``): joint/motor angles, ankle spring angle,
and SEA torques. Pure and stateless — no velocities, no filtering, no
control_enabled logic (those belong to the higher-level "Get Prosthesis State").

I/O is isolated in ``ethercat_io`` (stubbed until the pysoem layer lands), so the
same ``read_encoders()`` runs live on the bus and in offline replay/verification.
"""
from .ethercat_io import RawSensorFrame, read
from .read_encoders import EncoderCalibration, SensorReadings, CAL, read_encoders

__all__ = [
    "RawSensorFrame",
    "read",
    "EncoderCalibration",
    "SensorReadings",
    "CAL",
    "read_encoders",
]
