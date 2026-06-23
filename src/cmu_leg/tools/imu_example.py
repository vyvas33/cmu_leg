"""
Hands-on example for the vendored WitMotion driver — NOT the production wrapper.

Purpose: exercise the witapi surface end-to-end against ONE physical IMU so the
data (and the API lifecycle) is understood before writing the two-sensor adapter
(hardware/imu_io.py). Run it on the target with a single IMU plugged in:

    python -m cmu_leg.tools.imu_example

What it demonstrates, in order:
  1. discovery -> construct -> connect           (lifecycle)
  2. configure ranges / rate / LPF / outputs     (must happen BEFORE stream())
  3. stream() spawns the background parse thread  (non-blocking from here on)
  4. get_reading() returns the latest frame       (non-blocking; may repeat)
  5. freshness via pll_time_s                      (detect a genuinely new sample)
  6. a gravity sanity check                        (|acc| ~ 9.8 at rest)

Single sensor on purpose: discover_port() RAISES if two CH340 IMUs are plugged
in at once, so the two-sensor case needs explicit port paths — that's a job for
the wrapper, not this learning script.

lsusb will list: QinHeng Electronics HL-340 USB-Serial adapter

"""
import sys
import time
from math import sqrt

from cmu_leg.hardware.witmotion.witapi import WitMotionSensor, discover_port, discover_baud

# --- knobs (kept aligned with the baseline Simulink Read IMU block) ---
ACC_RANGE_G = 16        # Simulink scaled by 16/32768 g  -> ±16 g
GYRO_RANGE_DPS = 2000   # Simulink scaled by 2000/32768 dps -> ±2000 dps
STREAM_RATE_HZ = 100    # sensor output rate; raise later once baud headroom is confirmed
LPF_CUTOFF_HZ = 20      # anti-alias: keep <= 0.4 * rate (Nyquist guidance in the driver)
PRINT_HZ = 10           # throttle console prints; we still CONSUME every frame
DURATION_S = 30.0


def main():
    # 1) DISCOVERY + CONSTRUCT --------------------------------------------------
    # discover_port() scans for the CH340 USB-UART bridge (VID 6790). With a single
    # IMU connected it returns that port; with two it raises ValueError (the reason
    # the wrapper will pass explicit ports instead). discover_baud() probes each
    # supported rate and returns the one the sensor is actually talking at.
    try:
        port = discover_port()
        baud = discover_baud(port)
        sensor = WitMotionSensor(port=port, baud_bps=baud)
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"Discovery failed: {e}")
        print("If on macOS and the sensor is stuck at 921600 baud, run witrecovery.py.")
        sys.exit(1)

    try:
        # 2) CONNECT ------------------------------------------------------------
        # Opens the serial port, verifies comms with register reads (retries 10x),
        # and reads back the sensor's current rate/LPF/ranges into sensor.* attrs.
        # connect() does NOT start streaming and does NOT write config.
        sensor.connect()

        # 3) CONFIGURE (all @_config_only — must be BEFORE stream()) ------------
        # Setters read the register first and only write if the value differs, so
        # re-running this script doesn't needlessly hammer the flash.
        sensor.set_acc_range(ACC_RANGE_G)
        sensor.set_gyro_range(GYRO_RANGE_DPS)
        sensor.set_streaming_rate(STREAM_RATE_HZ)
        sensor.set_lpf(LPF_CUTOFF_HZ)
        # Only the outputs the ESKF needs: timestamp + accel + gyro. No mag/euler/
        # quat (the baseline filter is "noMag" and we run our own orientation).
        sensor.set_outputs(ts=True, acc=True, gyro=True, ang=False, mag=False, quat=False)

        print("Config:", sensor.get_data_header())

        # 4) STREAM -------------------------------------------------------------
        # Spawns the daemon parse thread and returns immediately. From here,
        # get_reading() is a non-blocking read of the latest committed frame.
        sensor.stream()

        # 5) CONSUME LOOP -------------------------------------------------------
        # get_reading() is non-destructive: it returns the SAME frame on repeated
        # calls until the parse thread overwrites it. We detect a genuinely new
        # sample by watching pll_time_s change — this is exactly the freshness
        # check the wrapper must carry into IMUFrame so the ESKF never integrates
        # a repeated (stale) sample.
        last_pll = None
        last_print = 0.0
        deadline = time.perf_counter() + DURATION_S
        try:
            while time.perf_counter() < deadline:
                data = sensor.get_reading()
                if data is not None and data["pll_time_s"] != last_pll:
                    last_pll = data["pll_time_s"]

                    a = data["acc"]    # [x, y, z] m/s², RAW SENSOR FRAME (no remap yet)
                    g = data["gyro"]   # [x, y, z] rad/s, RAW SENSOR FRAME

                    now = time.perf_counter()
                    if now - last_print >= 1.0 / PRINT_HZ:
                        last_print = now
                        # |acc| should sit near 9.81 when the sensor is held still;
                        # whichever axis is pointing up reads ~+9.8 (support force).
                        amag = sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
                        print(
                            f"{last_pll:7.3f}s | "
                            f"acc[{a[0]:+6.2f} {a[1]:+6.2f} {a[2]:+6.2f}] "
                            f"|a|={amag:5.2f} | "
                            f"gyro[{g[0]:+6.2f} {g[1]:+6.2f} {g[2]:+6.2f}]"
                        )

                # Oversample ~2x the sensor rate so we don't miss new frames.
                time.sleep(0.5 / sensor.rate)
        except KeyboardInterrupt:
            print("\nstopped early")

    except (ConnectionError, RuntimeError) as e:
        print(f"Error: {e}")
    finally:
        # 6) TEARDOWN — stops the parse thread and closes the port.
        sensor.disconnect()


if __name__ == "__main__":
    main()
