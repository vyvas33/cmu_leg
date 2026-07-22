# coding:UTF-8
"""
WitMotion HWT906 Sensor Settings Script.

Run this script to inspect and change the sensor's hardware settings. Two flags
at the top of the file control behaviour:

    SAVE_TO_FLASH          — prompt to persist all changes to the sensor's non-volatile
                             flash memory so they survive power cycles (default: False).
    ENABLE_CHANGE_BAUDRATE — gate the baud rate change, which can break communication
                             if set incorrectly (default: False).

Workflow:
    1. Edit the configuration values as needed in the try block below.
    2. Run 'python settings.py' to see the effect of the updated configuration.
    3. Set SAVE_TO_FLASH = True and rerun if you are ready to persist the
       changed configuration.

hgeyer, June 2026
"""

import sys
import time

from hwt906 import WitMotionSensor, discover_baud, discover_port

# Set True to allow changing the baud rate. Keep False by default — a wrong value
# can break communication and require running recovery.py to recover.
ENABLE_CHANGE_BAUDRATE = False


def main():
    print("---------------------------------------")
    print("WitMotion HWT906 Sensor Settings Script")
    print("---------------------------------------\n")

    print("Discovering sensor port and baudrate.")
    sensor = None
    try:
        port = discover_port()
        baud = discover_baud(port)
        sensor = WitMotionSensor(port=port, baud_bps=baud)
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # -------------------------------------------------------
    # Sensor configuration (saved to sensor flash if enabled)
    # -------------------------------------------------------
    # Edit the values below to match your setup. The driver only writes a register
    # if the requested value differs from the current hardware value, so methods
    # with unchanged values are no-ops.
    try:
        # Connect and restart sensor to force reloading registers from flash memory
        sensor.connect()
        sensor.reboot()
        sensor.disconnect()
        time.sleep(2.0)

        # Re-connect to sensor and show current config
        print("\nReconnecting ...\n")
        sensor.connect()

        print("\n\nCURRENT CONFIG IN SENSOR FLASH MEMORY:")
        sensor.show_config()  # print current hardware state before any changes

        # Serial baud rate (register 0x04)
        # Options: 9600, 115200, 230400, 460800, 921600 bps
        # NOTE: takes effect immediately; disconnect/reconnect required before saving.
        if ENABLE_CHANGE_BAUDRATE:
            sensor.set_baudrate(460800)
            sensor.disconnect()
            time.sleep(0.5)
            print("\nReconnecting ...\n")
            sensor.connect()

        # Streaming rate (register 0x03)
        # Options: 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 125, 200, 500, 1000 Hz
        sensor.set_streaming_rate(0.5)

        # Hardware analog low-pass filter cutoff (register 0x1F)
        # Options: 5, 10, 20, 42, 98, 188, 256 Hz
        # Rule of thumb: set to ~40% of streaming rate (e.g. 188 Hz at 500 Hz, 98 Hz at 200 Hz).
        # Pass filt_Hz=None above to let the driver apply this rule automatically.
        sensor.set_lpf(256)

        # Active output modalities (register 0x02)
        # Each flag enables the corresponding data packet in the stream.
        sensor.set_outputs(ts=False, acc=True, gyro=True, ang=False, mag=True, quat=False)

        # Gyroscope full-scale range (register 0x20)
        # Options: 250, 500, 1000, 2000 dps
        sensor.set_gyro_range(2000)

        # Accelerometer full-scale range (register 0x21)
        # Options: 2, 4, 8, 16 g
        sensor.set_acc_range(16)

        # Status LED (register 0x1B)
        sensor.set_led(enabled=False)

        # Mounting orientation (register 0x23)
        # vertical=False: flat/horizontal (default). vertical=True: upright.
        sensor.set_mounting_orientation(vertical=False)

        # Attitude fusion algorithm (register 0x24)
        # use_6axis=False: 9-axis — uses magnetometer to hold yaw (recommended outdoors).
        # use_6axis=True:  6-axis — yaw drifts; use when magnetometer is disturbed.
        #
        # HG Note June 2026: I get data drops during motion if the sensor uses 9 axis.
        # Not sure if this is consistent for all HWT906 or just a faulty sensor.
        # Setting the algorithm to 6axis=True completely resolves this issue.
        sensor.set_algorithm_axis(use_6axis=True)

        # Attitude filter smoothing coefficient (register 0x25)
        # Range: 1 (low latency, more noise) to 10000 (very smooth, high latency). Default: 50.
        # Affects Euler angle and quaternion output only — not raw acc/gyro.
        sensor.set_filter_coefficient(50)

        # Gyroscope auto-calibration (register 0x63)
        # Disable when running a custom ESKF bias estimator to avoid conflicts.
        sensor.set_gyro_autocal(enabled=False)

        print("SENSOR CONFIG AFTER CHANGES:")
        sensor.show_config()  # print updated hardware state

        if ENABLE_CHANGE_BAUDRATE:
            print("WARNING: Baud rate changed. Update any hardcoded baud_bps references accordingly.")
        if input("Persist settings to sensor flash? [Y/n] ").strip().upper() == "Y":
            sensor.save_config()

        # --------------------
        # Data collection test
        # --------------------
        print("\nDATA COLLECTION TEST:")
        print("\nData format:\n" + sensor.get_data_header() + "\n")

        # Set data rate and duration for data collection
        sample_rate = 4 * sensor.rate
        dt = 1 / sample_rate
        duration = 5 / sensor.rate  # at least 2s to cover thread startup and first-frame latency
        count = 0
        time_stamp = None

        # Enable streaming
        sensor.stream()

        data_stream = []

        start_time = clk = time.perf_counter()
        while time.perf_counter() - start_time <= duration:
            data = sensor.latest_data
            if data is not None and data["host_time_s"] != time_stamp:
                time_stamp = data["host_time_s"]
                count += 1
                data_stream.append((clk, data))

            clk += dt
            remaining = clk - time.perf_counter()
            time.sleep(max(max(remaining, 0), 0.001))

        for clk, data in data_stream:
            print(f"\nmain clk: {clk - start_time:.4f}")
            for k, v in data.items():
                print(f"{k}: {v}")

        print(f"\n{count} frames collected.\n")

    except (FileNotFoundError, ValueError, ConnectionError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    finally:
        if sensor is not None:
            sensor.disconnect()


if __name__ == "__main__":
    main()
