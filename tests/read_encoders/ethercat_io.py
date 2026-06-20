"""EtherCAT I/O boundary for read_encoders.

`RawSensorFrame` is the raw read result: integers exactly as they arrive on the
EtherCAT process-data objects (PDOs), before any scaling. It is the input of
`read_encoders()` — the seam that lets the same conversion code run live (frame
from pysoem) or offline (frame built from a recording), with no hardware needed
for the latter.

Field widths / signal sources mirror the ENI file
`Hardware/knee_ankle_prosthesis.xml`. The EtherCAT chain has three slaves in
this physical order: Knee Elmo (0), Ankle Elmo (1), Medulla 1.5 (2).

`unpack_frame()` is the pure byte->RawSensorFrame mapping (offline-testable).
`EtherCATInterface` owns the live pysoem master: bring the bus up with
`connect()`, then call `read()` each cycle. The Medulla streams its PDOs only
after DC SYNC0 is activated (`dc_sync`); without it the buffer stays all-zero.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import pysoem


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


# Slave indices on the EtherCAT chain (verified against the ENI file).
KNEE, ANKLE, MEDULLA = 0, 1, 2

# Medulla DC SYNC0 cycle time (ns) — matches ENI DC cycle time 0x000f4240.
MEDULLA_SYNC0_NS = 1_000_000


def unpack_frame(knee: bytes, ankle: bytes, medulla: bytes) -> RawSensorFrame:
    """Map the three slave input buffers into a `RawSensorFrame`.

    Pure: no hardware, no master — feed raw bytes (live or recorded) and get a
    frame back. Byte offsets/types come straight from the ENI PDO layout.

    Elmo (knee/ankle), 14 bytes: [0:4] position (int32), [8:10] status (uint16),
    [10:12] torque (int16), [12:14] analog (int16).
    Medulla, 23 bytes: [2] counter (uint8), [5:9] knee joint (uint32),
    [11:15] ankle joint (uint32), [17:19] hall0 (uint16), [19:21] hall1 (uint16).
    """
    return RawSensorFrame(
        knee_motor_counts=struct.unpack_from("<i", knee, 0)[0],
        knee_status_word=struct.unpack_from("<H", knee, 8)[0],
        knee_motor_torque_raw=struct.unpack_from("<h", knee, 10)[0],
        knee_analog=struct.unpack_from("<h", knee, 12)[0],
        ankle_motor_counts=struct.unpack_from("<i", ankle, 0)[0],
        ankle_status_word=struct.unpack_from("<H", ankle, 8)[0],
        ankle_motor_torque_raw=struct.unpack_from("<h", ankle, 10)[0],
        ankle_analog=struct.unpack_from("<h", ankle, 12)[0],
        medulla_counter=medulla[2],
        knee_joint_counts=struct.unpack_from("<I", medulla, 5)[0],
        ankle_joint_counts=struct.unpack_from("<I", medulla, 11)[0],
        hall_0=struct.unpack_from("<H", medulla, 17)[0],
        hall_1=struct.unpack_from("<H", medulla, 19)[0],
    )


def _elmo_pdo_config(slave) -> None:
    """SDO writes that map the Elmo TxPDOs (0x1a00/0x1a13/0x1a1d) and RxPDOs.

    Transcribed from the ENI [PS] init commands. Runs during `config_map()`.
    """
    # clear PDO assignments + mapping objects
    slave.sdo_write(0x1C12, 0, bytes([0x00]))
    slave.sdo_write(0x1C13, 0, bytes([0x00]))
    slave.sdo_write(0x1A07, 0, bytes([0x00]))
    slave.sdo_write(0x1A08, 0, bytes([0x00]))
    slave.sdo_write(0x1607, 0, bytes([0x00]))
    slave.sdo_write(0x1608, 0, bytes([0x00]))
    # assign RxPDOs to SM2 (0x1606/0x160b/0x160c)
    slave.sdo_write(0x1C12, 1, bytes([0x06, 0x16]))
    slave.sdo_write(0x1C12, 2, bytes([0x0B, 0x16]))
    slave.sdo_write(0x1C12, 3, bytes([0x0C, 0x16]))
    slave.sdo_write(0x1C12, 0, bytes([0x03]))
    # assign TxPDOs to SM3 (0x1a00/0x1a13/0x1a1d)
    slave.sdo_write(0x1C13, 1, bytes([0x00, 0x1A]))
    slave.sdo_write(0x1C13, 2, bytes([0x13, 0x1A]))
    slave.sdo_write(0x1C13, 3, bytes([0x1D, 0x1A]))
    slave.sdo_write(0x1C13, 0, bytes([0x03]))
    # Cyclic Sync Torque mode
    slave.sdo_write(0x6060, 0, bytes([0x0A]))


class EtherCATInterface:
    """Live EtherCAT master for the CMU leg: bring up the bus, then read frames.

    Usage:
        ec = EtherCATInterface("eth0")
        ec.connect()
        frame = ec.read()      # one RawSensorFrame per call
        ec.close()

    The drives stay un-enabled (controlword 0), so no torque is applied — this
    is a read-only sensor path even though the bus runs in OP.
    """

    def __init__(self, ifname: str = "eth0"):
        self._ifname = ifname
        self._master = pysoem.Master()

    def connect(self) -> None:
        try:
            self._bring_up()
        except Exception:
            # don't leave the master holding the NIC if bring-up fails
            self._master.close()
            raise

    def _bring_up(self) -> None:
        m = self._master
        m.open(self._ifname)
        if m.config_init() <= 0:
            raise RuntimeError("no EtherCAT slaves found")
        if len(m.slaves) != 3:
            raise RuntimeError(f"expected 3 slaves, found {len(m.slaves)}")

        m.slaves[KNEE].config_func = lambda pos: _elmo_pdo_config(m.slaves[pos])
        m.slaves[ANKLE].config_func = lambda pos: _elmo_pdo_config(m.slaves[pos])
        m.config_map()
        m.config_dc()
        m.slaves[MEDULLA].dc_sync(True, MEDULLA_SYNC0_NS)

        m.state = pysoem.SAFEOP_STATE
        m.write_state()
        if m.state_check(pysoem.SAFEOP_STATE, timeout=50_000) != pysoem.SAFEOP_STATE:
            raise RuntimeError(f"bus failed to reach SAFE_OP: {self._state_report()}")

        # exchange process data in SAFE_OP so the DC clocks lock before OP
        for _ in range(2000):
            m.send_processdata()
            m.receive_processdata(timeout=2000)

        m.state = pysoem.OP_STATE
        m.write_state()
        for _ in range(1000):
            m.send_processdata()
            m.receive_processdata(timeout=2000)
            if m.state_check(pysoem.OP_STATE, timeout=5_000) == pysoem.OP_STATE:
                break
        if m.state != pysoem.OP_STATE:
            raise RuntimeError(f"bus failed to reach OP: {self._state_report()}")

        # final settle so the Medulla is streaming before the first read
        for _ in range(2000):
            m.send_processdata()
            m.receive_processdata(timeout=2000)

    def _state_report(self) -> str:
        self._master.read_state()
        return ", ".join(
            f"{s.name}: state={s.state} al=0x{s.al_status:04x}"
            for s in self._master.slaves
        )

    def read(self) -> RawSensorFrame:
        """Exchange one PDO cycle and return the unpacked frame."""
        m = self._master
        m.send_processdata()
        m.receive_processdata(timeout=2000)
        return unpack_frame(
            bytes(m.slaves[KNEE].input),
            bytes(m.slaves[ANKLE].input),
            bytes(m.slaves[MEDULLA].input),
        )

    def close(self) -> None:
        try:
            self._master.state = pysoem.INIT_STATE
            self._master.write_state()
        finally:
            self._master.close()
