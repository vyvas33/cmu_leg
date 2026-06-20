"""EtherCAT I/O boundary for read_encoders.

`RawSensorFrame` is the raw read result: integers exactly as they arrive on the
EtherCAT process-data objects (PDOs), before any scaling. It is the *output* of
`read()` and the *input* of `read_encoders()` — the seam that lets the same
conversion code run live (frame from pysoem) or offline (frame built from a
recording), with no hardware needed for the latter.

Field widths / signal sources mirror the ENI file
`Hardware/knee_ankle_prosthesis.xml`. The EtherCAT chain has three slaves:
Medulla 1.5 (joint encoders, Hall GRF, status/counter), Knee Elmo, Ankle Elmo.

`read()` is a stub until the pysoem layer is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RawSensorFrame:
    """Raw integer PDO signals for one EtherCAT cycle.

    Only the four count fields are needed for the angle/torque conversions;
    the rest are carried for completeness (and for the future full
    "Get Prosthesis State"). All default to 0 so a replay harness can populate
    just the signals a given recording contains.
    """

    # --- Knee Elmo (TxPDO 0x1a00 / 0x1a13 / 0x1a1d) ---
    knee_motor_counts: int = 0      # Position actual value (DINT, 2^30 abs encoder)
    knee_status_word: int = 0       # Status word (UINT, CiA-402; ==15 -> enabled)
    knee_motor_torque_raw: int = 0  # Torque actual value (INT)
    knee_analog: int = 0            # Analog Input 1 (INT) -> PT1000 temp

    # --- Ankle Elmo (same layout) ---
    ankle_motor_counts: int = 0     # Position actual value (DINT, 2^30 abs encoder)
    ankle_status_word: int = 0      # Status word (UINT)
    ankle_motor_torque_raw: int = 0 # Torque actual value (INT)
    ankle_analog: int = 0           # Analog Input 1 (INT) -> PT1000 temp

    # --- Medulla 1.5 (TxPDO 0x1a00..0x1a05) ---
    medulla_counter: int = 0        # Counter (USINT); ticks each cycle when alive
    knee_joint_counts: int = 0      # Knee Encoder (UDINT, 17-bit joint absolute)
    ankle_joint_counts: int = 0     # Ankle Encoder (UDINT, pre-wrap raw)
    hall_0: int = 0                 # Hall 0 (UINT) -> toe GRF
    hall_1: int = 0                 # Hall 1 (UINT) -> heel GRF


def read() -> RawSensorFrame:
    """Read one cycle of PDOs off the EtherCAT bus.

    Stub: the pysoem master / PDO unpacking is not implemented yet. The live
    implementation will map the ENI byte layout into a `RawSensorFrame`.
    """
    raise NotImplementedError(
        "EtherCAT read() not implemented — pysoem master/PDO layer is a stub. "
        "Build a RawSensorFrame directly (e.g. from a recording) for offline use."
    )
