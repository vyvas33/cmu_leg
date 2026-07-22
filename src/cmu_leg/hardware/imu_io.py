"""
IMU Communication Interface for two Witmotion IMUs over USB.
This wrapper is built on custom WitMotion API and mirrors the EtherCAT interface.
However, unlike the ec.read() which outputs unscaled raw counts,
imu.read() here outputs in SI units in sensors' frame.
Typical Usage:

discover_imu_ports()                            # list CH340 ports and manually set params.IMU
tb, fb = discover_imu_bauds()                   # probe each IMU's current baud
imu = IMUInterface(thigh_baud=tb, foot_baud=fb) # ports from params, baud passed in
imu.connect()
frame = imu.read() # one frame per call; non-blocking
imu.close()
"""
import serial.tools.list_ports

from cmu_leg.config.params import IMU
from cmu_leg.hardware.witmotion.hwt906 import WitMotionSensor, discover_baud
from cmu_leg.types.sensors import IMUFrame

_CH340_VID = 6790  # CH340 / QinHeng USB-serial bridge (the chip hwt906 scans for)


def discover_imu_ports() -> list[tuple[str, str | None, str | None]]:
    """List connected CH340 (WitMotion) serial ports as (device, serial, location).
    """
    return [
        (p.device, p.serial_number, p.location)
        for p in serial.tools.list_ports.comports()
        if p.vid == _CH340_VID
    ]


def discover_imu_bauds(thigh_port: str | None = None, foot_port: str | None = None) -> tuple[int, int]:
    """Probe each IMU's current baud rate (per port). Ports default to params.IMU.
    Returns (thigh_baud, foot_baud) — pass straight into IMUInterface()."""
    return (
        discover_baud(thigh_port or IMU.thigh_port),
        discover_baud(foot_port or IMU.foot_port),
    )

def _connect_sensor(sensor: WitMotionSensor, attempts: int = 3) -> None:
    """Connect one sensor, retrying on flaky register reads.

    WitMotionSensor.connect() verifies comms with a retry loop but then does
    single-shot register reads to sync hardware state; on a marginal link those
    can time out with no retry. Retry the whole connect, closing the half-open
    port between attempts so the next try starts clean."""
    for attempt in range(1, attempts + 1):
        try:
            sensor.connect()
            return
        except (TimeoutError, ConnectionError):
            sensor.disconnect()
            if attempt == attempts:
                raise


def _configure(sensor: WitMotionSensor) -> None:
    """Apply standard config to one connected sensor, then start streaming"""
    sensor.set_acc_range(IMU.acc_range_g)
    sensor.set_gyro_range(IMU.gyro_range_dps)
    sensor.set_streaming_rate(IMU.stream_rate_hz)
    sensor.set_lpf(IMU.lpf_cutoff_hz)
    sensor.set_outputs(ts=True, acc=True, gyro=True, ang=False, mag=False, quat=False)
    sensor.stream()
    
class IMUInterface:
    def __init__(self, thigh_port: str | None = None, foot_port: str | None = None,
                 thigh_baud: int | None = None, foot_baud: int | None = None):
        # default to params ports/baud; pass explicit values to override
        # (e.g. thigh_baud/foot_baud from discover_imu_bauds()).
        self.thigh = WitMotionSensor(port=thigh_port or IMU.thigh_port, baud_bps=thigh_baud or IMU.baud)
        self.foot = WitMotionSensor(port=foot_port or IMU.foot_port, baud_bps=foot_baud or IMU.baud)

        # last host_time_s to drive the freshness flag
        self._last_thigh_t = None
        self._last_foot_t = None

    def connect(self) -> None:
        try:
            _connect_sensor(self.thigh)
            _connect_sensor(self.foot)
            _configure(self.thigh)
            _configure(self.foot)
        except Exception:
            self.close()
            raise

    def read(self) -> IMUFrame:
        # latest_data is a non-destructive zero-order hold: it repeats the last
        # frame until the parser thread overwrites it, hence the freshness flags.
        th = self.thigh.latest_data
        fo = self.foot.latest_data

        if th is None:
            thigh_acc = (0.0, 0.0, 0.0)
            thigh_gyro = (0.0, 0.0, 0.0)
            thigh_time, thigh_fresh = 0.0, False
        
        else:
            thigh_acc = tuple(th["acc"])
            thigh_gyro = tuple(th["gyro"])
            thigh_time = th["host_time_s"]
            thigh_fresh = True if thigh_time != self._last_thigh_t else False
            self._last_thigh_t = thigh_time

        if fo is None:
            foot_acc = (0.0, 0.0, 0.0)
            foot_gyro = (0.0, 0.0, 0.0)
            foot_time, foot_fresh = 0.0, False
        
        else:
            foot_acc = tuple(fo["acc"])
            foot_gyro = tuple(fo["gyro"])
            foot_time = fo["host_time_s"]
            foot_fresh = True if foot_time != self._last_foot_t else False
            self._last_foot_t = foot_time

        return IMUFrame(
            thigh_acc=thigh_acc,
            thigh_gyro=thigh_gyro,
            foot_acc=foot_acc,
            foot_gyro=foot_gyro,
            thigh_time=thigh_time,
            foot_time=foot_time,
            thigh_fresh=thigh_fresh,
            foot_fresh= foot_fresh)
    
    def close(self) -> None:
        try: 
            self.thigh.disconnect()
        finally:
            self.foot.disconnect()


