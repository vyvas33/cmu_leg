"""
IMU Communication Interface for two Witmotion IMUs over USB.
This wrapper is built on custom WitMotion API and mirrors the EtherCAT interface.
However, unlike the ec.read() which outputs unscaled raw counts,
imu.read() here outputs in SI units in sensors' frame.
Typical Usage:

discover_imu_ports()       # list CH340 and set params manually
imu = IMUInterface()       
imu.connect()
frame = imu.read() # one frame per call; non-blocking
imu.close()
"""
import serial.tools.list_ports

from cmu_leg.config.params import IMU
from cmu_leg.hardware.witmotion.witapi import WitMotionSensor
from cmu_leg.types.sensors import IMUFrame

_CH340_VID = 6790  # CH340 / QinHeng USB-serial bridge (the chip witapi scans for)


def discover_imu_ports() -> list[tuple[str, str | None, str | None]]:
    """List connected CH340 (WitMotion) serial ports as (device, serial, location).
    """
    return [
        (p.device, p.serial_number, p.location)
        for p in serial.tools.list_ports.comports()
        if p.vid == _CH340_VID
    ]

def _configure(sensor: WitMotionSensor) -> None:
    """Apply standard config to one connected sensor, then start streaming"""
    sensor.set_acc_range(IMU.acc_range_g)
    sensor.set_gyro_range(IMU.gyro_range_dps)
    sensor.set_streaming_rate(IMU.stream_rate_hz)
    sensor.set_lpf(IMU.lpf_cutoff_hz)
    sensor.set_outputs(ts=True, acc=True, gyro=True, ang=False, mag=False, quat=False)
    sensor.stream()
    
class IMUInterface:
    def __init__(self, thigh_port: str | None = None, foot_port: str | None = None):
        # default to the params ports; pass explicit paths to override
        self.thigh = WitMotionSensor(port=thigh_port or IMU.thigh_port, baud_bps=IMU.baud)
        self.foot = WitMotionSensor(port=foot_port or IMU.foot_port, baud_bps=IMU.baud)

        # last pll_time_s to drive the freshness flag  
        self._last_thigh_t = None
        self._last_foot_t = None 

    def connect(self) -> None:
        try: 
            self.thigh.connect()
            self.foot.connect()
            _configure(self.thigh)
            _configure(self.foot)
        except Exception:
            self.close()
            raise

    def read(self) -> IMUFrame:
        th = self.thigh.get_reading()
        fo = self.foot.get_reading()

        if th is None:
            thigh_acc = (0.0, 0.0, 0.0)
            thigh_gyro = (0.0, 0.0, 0.0)
            thigh_time, thigh_fresh = 0.0, False
        
        else:
            thigh_acc = tuple(th["acc"])
            thigh_gyro = tuple(th["gyro"])
            thigh_time = th["pll_time_s"]
            thigh_fresh = True if thigh_time != self._last_thigh_t else False
            self._last_thigh_t = thigh_time

        if fo is None:
            foot_acc = (0.0, 0.0, 0.0)
            foot_gyro = (0.0, 0.0, 0.0)
            foot_time, foot_fresh = 0.0, False
        
        else:
            foot_acc = tuple(fo["acc"])
            foot_gyro = tuple(fo["gyro"])
            foot_time = fo["pll_time_s"]
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


