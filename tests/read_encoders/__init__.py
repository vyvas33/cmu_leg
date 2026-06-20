"""Python port of the Simulink ``read_encoders`` model.

Turns raw EtherCAT PDO integers (``RawSensorFrame``) into engineering-unit
sensor readings (``SensorReadings``): joint/motor angles, ankle spring angle,
and SEA torques. Pure and stateless — no velocities, no filtering, no
control_enabled logic (those belong to the higher-level "Get Prosthesis State").

I/O is isolated in ``ethercat_io`` (``EtherCATInterface`` for the live bus,
``unpack_frame`` as the pure byte->frame seam), so the same ``read_encoders()``
runs live on the bus and in offline replay/verification.
"""
from .ethercat_io import RawSensorFrame, EtherCATInterface, unpack_frame
from .read_encoders import EncoderCalibration, SensorReadings, CAL, read_encoders

__all__ = [
    "RawSensorFrame",
    "EtherCATInterface",
    "unpack_frame",
    "EncoderCalibration",
    "SensorReadings",
    "CAL",
    "read_encoders",
]
