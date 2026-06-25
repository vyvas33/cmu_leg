"""
EtherCAT Communication Interface.
Input: Slave to Master, Output: Master to Slave
Usage:
    ec = EtherCATInterface("eth0")
    ec.connect()
    frame = ec.read()      # one RawEtherCATFrame per call
    ec.close()

vkalva, June 2026
"""
import struct
import pysoem
from cmu_leg.types.sensors import RawEtherCATFrame
from cmu_leg.types.commands import DriveCommand
from cmu_leg.hardware.elmos.state_machine import drive_state, DriveState, control_word

KNEE, ANKLE, MEDULLA = 0, 1, 2 # slave indices on the EtherCAT Chain

MEDULLA_SYNC0_NS = 1_000_000 # SYNC0 period = ENI DC cycle time (0x000f4240); must match the loop rate

# Ticks the medulla counter may repeat before we treat the medulla as dead. The
# medulla increments its counter every cycle, so a stalled counter means the board
# (or the bus) has stopped cycling. Tune on hardware.
MEDULLA_STALL_LIMIT = 5

def unpack_frame(knee: bytes, ankle: bytes, medulla: bytes) -> RawEtherCATFrame:
    """Map the three slave input buffers to a RawEtherCATFrame. Offsets/types per
    README.md. Pure: feed live (slave.input) or recorded bytes."""
    return RawEtherCATFrame(
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

def _pack_drive(velocity_counts: int, torque_per_mille: int, cw: int) -> bytes:
    """Pack one Elmo's 24-byte output PDO (offsets per README; unused fields stay 0).
    Output-side mirror of unpack_frame. Mode is fixed to CSV (9)."""
    buf = bytearray(24)
    struct.pack_into("<i", buf, 8, velocity_counts)    # 0x60ff Target Velocity
    struct.pack_into("<h", buf, 16, torque_per_mille)  # 0x60b2 Torque Offset
    struct.pack_into("<H", buf, 18, cw)                 # 0x6040 Control word
    struct.pack_into("<b", buf, 20, 9)                  # 0x6060 Mode of operation = CSV
    return bytes(buf)

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

        self.enable = False                    # master switch; the main loop sets this
        self._last_frame = RawEtherCATFrame()  # latest inputs, for the readiness checks
        self._prev_medulla_counter = None      # previous counter, for the liveness check
        self._medulla_stall_ticks = 0          # consecutive ticks the counter hasn't moved

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

    def read(self) -> RawEtherCATFrame:
        m = self.master
        m.send_processdata()
        m.receive_processdata(timeout=2000)
        frame = unpack_frame(
            bytes(m.slaves[KNEE].input),
            bytes(m.slaves[ANKLE].input),
            bytes(m.slaves[MEDULLA].input),
        )
        self._ingest(frame)
        return frame

    def _ingest(self, frame: RawEtherCATFrame) -> None:
        """Stash the latest frame and track medulla liveness. Called by every
        exchange (read and, later, write)."""
        if frame.medulla_counter == self._prev_medulla_counter:
            self._medulla_stall_ticks += 1
        else:
            self._medulla_stall_ticks = 0
        self._prev_medulla_counter = frame.medulla_counter
        self._last_frame = frame

    def drives_enabled(self) -> bool:
        """Both Elmos report CiA-402 Operation Enabled (the status word, not the
        commanded control word)."""
        return (
            drive_state(self._last_frame.knee_status_word) == DriveState.OPERATION_ENABLED
            and drive_state(self._last_frame.ankle_status_word) == DriveState.OPERATION_ENABLED
        )

    def medulla_alive(self) -> bool:
        """Medulla counter is still advancing (the board and bus are cycling)."""
        return self._medulla_stall_ticks < MEDULLA_STALL_LIMIT

    def control_enabled(self) -> bool:
        """Readiness, recomputed each tick: both drives enabled and medulla alive.
        The main loop reads this to decide whether to run the controllers."""
        return self.drives_enabled() and self.medulla_alive()

    def write(self, cmd: DriveCommand) -> RawEtherCATFrame:
        """One output cycle: derive each Elmo's control word from its last status
        word and the master `enable`, pack the command plus control word into the
        output PDOs, exchange, and return the fresh input frame. Call every tick.

        The control word is computed from the previous frame's status word (the
        inherent one-cycle latency), then this cycle's inputs are read back in."""
        m = self.master
        f = self._last_frame

        knee_cw = control_word(f.knee_status_word, self.enable)
        ankle_cw = control_word(f.ankle_status_word, self.enable)
        m.slaves[KNEE].output = _pack_drive(cmd.knee_velocity_counts, cmd.knee_torque_per_mille, knee_cw)
        m.slaves[ANKLE].output = _pack_drive(cmd.ankle_velocity_counts, cmd.ankle_torque_per_mille, ankle_cw)

        # The baseline never writes the medulla output PDO (no PDO Transmit targets the
        # medulla), so we leave it at default -- the medulla does not watchdog a master
        # heartbeat. We only read its input counter (medulla_alive).

        m.send_processdata()
        m.receive_processdata(timeout=2000)
        frame = unpack_frame(
            bytes(m.slaves[KNEE].input),
            bytes(m.slaves[ANKLE].input),
            bytes(m.slaves[MEDULLA].input),
        )
        self._ingest(frame)
        return frame

    def close(self) -> None:
        try:
            self.master.state = pysoem.INIT_STATE
            self.master.write_state()
        finally:
            self.master.close()
