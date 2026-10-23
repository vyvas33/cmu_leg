"""
Bertec Treadmill Control Panel -- Remote TCP/IP Control client.
Python port of HMRL-Matlab-Treadmill-Functions (Pablo Iturralde, 2013-2015)
https://github.com/willpower2727/HMRL-Matlab-Treadmill-Functions
(getPayload.m, int16toBytes.m, readTreadmillPacket.m, setTreadmill.m).

Like imu_io, read() returns SI units (m/s, deg); the wire carries mm/s and
hundredths of a degree. Talks only to the Control Panel socket, NOT to
BertecDevice.dll (the separate force-plate SDK).

Setup, on the prosthesis computer (Windows):
  1. Open the Treadmill Control Panel (Bertec treadmill logo).
  2. Settings: enable Remote TCP/IP Control and uncheck "Listen on 127.0.0.1
     Only".
  3. Hit "Ready for Remote Control" on the dashboard. Without it status still
     streams but commands are silently ignored.
  4. Then run the code on the Jetson (see bertec_test.py).

Typical Usage:

tm = BertecTreadmill()              # host/port from params.TREADMILL
tm.connect()
frame = tm.read()                   # latest status; non-blocking
tm.set_speeds(0.5, 0.5)             # m/s, right then left
tm.stop()
tm.close()                          # does NOT stop the belts; stop() first

Verified on hardware (Oct 2026): status read with belts run by hand, and a
0.2 m/s command at 0.3 m/s^2 accel. Not yet checked: which belt is right vs
left, and the sign of "forward".

vkalva, Claude, October 2026
"""
import socket
import struct
import time
from dataclasses import replace

from cmu_leg.config.params import TREADMILL
from cmu_leg.structures.sensors import TreadmillFrame

# Limits from getPayload.m, in wire units
MAX_ACC_MM_S2 = 3000
DEFAULT_ACC_MM_S2 = 1000    # substituted for a non-positive accel (getPayload.m)
MAX_VEL_MM_S = 6500
MIN_VEL_MM_S = -6500
MAX_INCLINE_CDEG = 1500     # hundredths of a degree (setTreadmill.m) -> 15.0 deg

COMMAND_SIZE = 64           # format + 9 int16 + 18 check bytes + 27 padding
STATUS_SIZE = 32            # format + 4 int16 speeds + int16 incline + 21 padding


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def pack_command(speed_r_mm_s: int, speed_l_mm_s: int, acc_r_mm_s2: int,
                 acc_l_mm_s2: int, incline_cdeg: int = 0) -> bytes:
    """Build the 64-byte command payload, as getPayload.m does:

      byte 0      : format = 0
      bytes 1-18  : 9 big-endian int16
                    [speedR, speedL, 0, 0, accR, accL, 0, 0, incline]
      bytes 19-36 : 255 - each data byte (redundancy check)
      bytes 37-63 : zero padding

    Pure: inputs in wire units, already rounded to int."""
    # getPayload.m's "accR<1" test is in mm/s^2, i.e. it replaces only a
    # zero/negative accel with the default. It is NOT a 1 m/s^2 floor.
    if acc_r_mm_s2 < 1:
        acc_r_mm_s2 = DEFAULT_ACC_MM_S2
    if acc_l_mm_s2 < 1:
        acc_l_mm_s2 = DEFAULT_ACC_MM_S2

    values = (
        _clamp(speed_r_mm_s, MIN_VEL_MM_S, MAX_VEL_MM_S),
        _clamp(speed_l_mm_s, MIN_VEL_MM_S, MAX_VEL_MM_S),
        0, 0,
        min(acc_r_mm_s2, MAX_ACC_MM_S2),
        min(acc_l_mm_s2, MAX_ACC_MM_S2),
        0, 0,
        _clamp(incline_cdeg, 0, MAX_INCLINE_CDEG),
    )
    data = struct.pack(">9h", *values)          # MSB first, as int16toBytes
    check = bytes(255 - b for b in data)
    return bytes([0]) + data + check + bytes(27)


def unpack_status(packet: bytes) -> tuple[int, int, int]:
    """Map one 32-byte status packet to (speed_r_mm_s, speed_l_mm_s, incline_cdeg),
    per readTreadmillPacket.m. Byte 0 is the format; speeds 3/4 are unused."""
    speed_r, speed_l, _rr, _ll, incline = struct.unpack_from(">5h", packet, 1)
    return speed_r, speed_l, incline


class BertecTreadmill:
    def __init__(self, host: str | None = None, port: int | None = None,
                 timeout_s: float | None = None):
        # default to params; pass explicit values to override
        self.host = host or TREADMILL.host
        self.port = port or TREADMILL.port
        self.timeout_s = timeout_s or TREADMILL.timeout_s
        self.sock: socket.socket | None = None

        # TCP is a byte stream: a recv() can end mid-packet, so carry the
        # remainder over instead of dropping it (which would misalign every
        # packet after it).
        self._rx = bytearray()
        self._last = TreadmillFrame()

    def connect(self) -> None:
        if not self.host:
            raise ValueError("No treadmill host: set params.TREADMILL.host or pass host=")
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # send each command now
        self._rx.clear()
        self._last = TreadmillFrame()

    def read(self) -> TreadmillFrame:
        """Drain the socket and return the newest complete status packet.
        Non-blocking: with nothing new, repeats the last frame with fresh=False
        (all zeros before the first packet)."""
        if self.sock is None:
            raise RuntimeError("Not connected; call connect() first")

        # Non-blocking only for the drain. MSG_DONTWAIT is not enough: on a
        # socket with a timeout Python polls for data BEFORE recv, so it would
        # still block. Restore the timeout so sendall() keeps it.
        self.sock.settimeout(0.0)
        try:
            while True:
                try:
                    chunk = self.sock.recv(4096)
                except BlockingIOError:
                    break
                if not chunk:
                    raise ConnectionError("Treadmill Control Panel closed the connection")
                self._rx += chunk
        finally:
            self.sock.settimeout(self.timeout_s)

        n = len(self._rx) // STATUS_SIZE
        if n == 0:
            # a copy, so a frame the caller already holds doesn't change under it
            return replace(self._last, fresh=False)

        # keep only the newest whole packet; the partial tail stays buffered
        packet = bytes(self._rx[(n - 1) * STATUS_SIZE : n * STATUS_SIZE])
        del self._rx[: n * STATUS_SIZE]

        speed_r, speed_l, incline = unpack_status(packet)
        self._last = TreadmillFrame(
            right_speed=speed_r / 1000.0,
            left_speed=speed_l / 1000.0,
            incline=incline / 100.0,
            time=time.monotonic(),
            fresh=True,
        )
        return self._last

    def set_speeds(self, speed_r: float, speed_l: float,
                   acc_r: float = 1.0, acc_l: float = 1.0,
                   incline: float = 0.0) -> None:
        """Command belt speeds [m/s] and the accelerations [m/s^2] used to reach
        them. Out-of-range values are clamped as in getPayload.m; a non-positive
        accel becomes 1 m/s^2. incline [deg], 0..15."""
        if self.sock is None:
            raise RuntimeError("Not connected; call connect() first")
        self.sock.sendall(pack_command(
            speed_r_mm_s=round(speed_r * 1000),
            speed_l_mm_s=round(speed_l * 1000),
            acc_r_mm_s2=round(acc_r * 1000),
            acc_l_mm_s2=round(acc_l * 1000),
            incline_cdeg=round(incline * 100),
        ))

    def stop(self, acc: float = 1.0) -> None:
        """Ramp both belts to zero at acc [m/s^2]. Leaves incline at 0."""
        self.set_speeds(0.0, 0.0, acc_r=acc, acc_l=acc)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
