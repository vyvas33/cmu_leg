import sys, time
from cmu_leg.hardware.witmotion.witapi import WitMotionSensor, discover_port, discover_baud

sensor = None
try:
    port = "/dev/ttyUSB0"  # replace with port name if known
    baud = discover_baud(port)  # replace with sensor baud rate if known
    sensor = WitMotionSensor(port=port, baud_bps=baud)
except (FileNotFoundError, ValueError, ConnectionError) as e:
    print(f"Error: {e}"); sys.exit(1)

try:
    sensor.connect()
    sensor.set_streaming_rate(50)
    sensor.set_lpf(20)  # ensure cutoff frequency <= 0.4 * stream_rate to avoid aliasing
    sensor.set_outputs(ts=True, acc=True, gyro=True, ang=False, mag=False, quat=False)

    print(sensor.get_data_header())  # convenience function

    sensor.stream()

    last_ts = None
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        data = sensor.get_reading()
        if data is not None and data["pll_time_s"] != last_ts:
            last_ts = data["pll_time_s"]

            a, g = data["acc"], data["gyro"]
            print(f"{last_ts:.3f}s | [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}] m/s²")

        time.sleep(0.5 / sensor.rate)  # oversampling by at least 2x to catch all new data

except (ConnectionError, RuntimeError) as e:
    print(f"Error: {e}")
finally:
    if sensor is not None:
        sensor.disconnect()