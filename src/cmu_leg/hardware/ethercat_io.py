"""
EtherCAT Communication Interface.
Input: Slave to Master, Output: Master to Slave

Usage:
    ec = EtherCATInterface("eth0")
    ec.connect()
    frame = ec.read()      # one RawSensorFrame per call
    ec.close()

"""

import struct
from dataclasses import dataclass

import pysoem

KNEE, ANKLE, MEDULLA = 0, 1, 2 # slave indices on the EtherCAT Chain

MEDULLA_SYNC0_NS = 1_000_000 # need to verify this, can we change this?


@dataclass(slots=True)
class RawSensorFrame:
    """Raw integer PDO inputs for one EtherCAT cycle. See README.md for the
    byte/type/source map. Defaults let a replay harness build partial frames."""

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


def unpack_frame(knee: bytes, ankle: bytes, medulla: bytes) -> RawSensorFrame:
    """Map the three slave input buffers to a RawSensorFrame. Offsets/types per
    README.md. Pure: feed live (slave.input) or recorded bytes."""
    return RawSensorFrame(
        knee_motor_counts=struct.unpack_from("<i", knee, 0)[0],
        knee_digital_inputs=struct.unpack_from("<I", knee, 4)[0],
        knee_status_word=struct.unpack_from("<H", knee, 8)[0],
        knee_motor_torque_raw=struct.unpack_from("<h", knee, 10)[0],
        knee_analog=struct.unpack_from("<h", knee, 12)[0],

        ankle_motor_counts=struct.unpack_from("<i", ankle, 0)[0],
        ankle_digital_inputs=struct.unpack_from("<I", ankle, 4)[0],
        ankle_status_word=struct.unpack_from("<H", ankle, 8)[0],
        ankle_motor_torque_raw=struct.unpack_from("<h", ankle, 10)[0],
        ankle_analog=struct.unpack_from("<h", ankle, 12)[0],

        medulla_id=struct.unpack_from("<B", medulla, 0)[0],
        medulla_state=struct.unpack_from("<B", medulla, 1)[0],
        medulla_counter=struct.unpack_from("<B", medulla, 2)[0],
        medulla_error_flags=struct.unpack_from("<B", medulla, 3)[0],
        medulla_limit_switch=struct.unpack_from("<B", medulla, 4)[0],
        knee_joint_counts=struct.unpack_from("<I", medulla, 5)[0],
        knee_joint_timestamp=struct.unpack_from("<H", medulla, 9)[0],
        ankle_joint_counts=struct.unpack_from("<I", medulla, 11)[0],
        ankle_joint_timestamp=struct.unpack_from("<H", medulla, 15)[0],
        hall_0=struct.unpack_from("<H", medulla, 17)[0],
        hall_1=struct.unpack_from("<H", medulla, 19)[0],
        medulla_logic_voltage=struct.unpack_from("<H", medulla, 21)[0],
    )


# Elmo PDO assignment (see README): RxPDOs -> SM2 (0x1C12), TxPDOs -> SM3 (0x1C13).
ELMO_RXPDOS = (0x1606, 0x160B, 0x160C)
ELMO_TXPDOS = (0x1A00, 0x1A13, 0x1A1D)
ELMO_OP_MODE = 0x0A  # ENI startup default (CST); runtime mode is CSV (9), written via PDO byte 20


def _assign_pdos(slave, assign_obj: int, pdos: tuple[int, ...]) -> None:
    """Set a Sync Manager PDO assignment: clear count, write entries, set count."""
    slave.sdo_write(assign_obj, 0, bytes([0]))
    for i, pdo in enumerate(pdos, start=1):
        slave.sdo_write(assign_obj, i, struct.pack("<H", pdo))
    slave.sdo_write(assign_obj, 0, bytes([len(pdos)]))


def configure_elmo(slave) -> None:
    """PreOp->SafeOp CoE config for one Elmo. Used as a pysoem config_func.
    Verified against the ENI init commands and nm_nn_swing (Get Prosthesis State)."""
    _assign_pdos(slave, 0x1C12, ELMO_RXPDOS)       # outputs -> SM2
    _assign_pdos(slave, 0x1C13, ELMO_TXPDOS)       # inputs  -> SM3
    slave.sdo_write(0x6060, 0, bytes([ELMO_OP_MODE]))  # op mode
    slave.sdo_write(0x60C2, 1, bytes([1]))             # interpolation time period = 1 (ms)

class EtherCATInterface:

    def __init__(self, ifname: str = "eth0"):
        self.ifname = ifname
        self.master = pysoem.Master()

    def connect(self) -> None:
        try:
            self._bring_up()
            print("Connected")
        except Exception:
            self.master.close()
            raise

    def _bring_up(self) -> None:
        m = self.master
        m.open(self.ifname) # will return an error if not found

        # check if slaves are found
        if m.config_init() <= 0:
            raise RuntimeError("No EtherCAT slaves found")
        if len(m.slaves) != 3:
            raise RuntimeError(f"Expected 3 slaves, found {len(m.slaves)}")
        
        # set config functions
        m.slaves[KNEE].config_func = lambda pos: configure_elmo(m.slaves[pos])
        m.slaves[ANKLE].config_func = lambda pos: configure_elmo(m.slaves[pos])

        # write configs 
        m.config_map()
        m.config_dc()
        m.slaves[MEDULLA].dc_sync(True, MEDULLA_SYNC0_NS)

        m.state = pysoem.SAFEOP_STATE
        m.write_state()
        if m.state_check(pysoem.SAFEOP_STATE, timeout=50_000) != pysoem.SAFEOP_STATE:
            raise RuntimeError(f"bus failed to reach SAFE_OP: {self._state_report()}")
        
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
        
        for _ in range(2000):
            m.send_processdata()
            m.receive_processdata(timeout=2000)

    def _state_report(self) -> str:
        self.master.read_state()
        return ", ".join(
            f"{s.name}: state={s.state} al=0x{s.al_status:04x}"
            for s in self.master.slaves
        )

    def read(self):
        m = self.master
        m.send_processdata()
        m.receive_processdata(timeout=2000)
        return unpack_frame(
            bytes(m.slaves[KNEE].input),
            bytes(m.slaves[ANKLE].input),
            bytes(m.slaves[MEDULLA].input),
        )
    
    def close(self) -> None:
        try:
            self.master.state = pysoem.INIT_STATE
            self.master.write_state()
        finally:
            self.master.close()
