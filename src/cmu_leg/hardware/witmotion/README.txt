================================================================================
WITMOTION HWT906 PYTHON DRIVER API
================================================================================

This README describes the architecture, configuration features, and programming
interface for the custom, single-file Python driver `WitMotionSensor` located in
api/witapi.py.

This API is designed to make connecting to, configuring, and receiving data from
the WitMotion HWT906 sensor plug-and-play, even for users with rudimentary
knowledge of serial interfaces, microcontrollers, and inertial sensors.

hgeyer, Jun 2026

--------------------------------------------------------------------------------
1. INTRODUCTION: WHAT IS THE HWT906 SENSOR?
--------------------------------------------------------------------------------
The WitMotion HWT906 is an Inertial Measurement Unit (IMU) and Attitude and
Heading Reference System (AHRS). It contains:
  * Accelerometer: Measures linear acceleration along 3 axes (X, Y, Z).
  * Gyroscope: Measures angular rotation speed (pitch, roll, yaw rates) on 3 axes.
  * Magnetometer: Measures the surrounding magnetic field on 3 axes (for heading).
  * Temperature Sensor: Measures the sensor die temperature in °C. Reported alongside
    accelerometer data (present when acc output is enabled).
  * Onboard Processor: Merges these raw signals to compute a drift-free orientation
    vector (Euler angles or Quaternions) relative to the Earth.

This driver establishes a connection to the sensor over USB (via a virtual serial port),
configures its hardware parameters, and streams clean data directly to your code.

--------------------------------------------------------------------------------
2. CANNOT ESTABLISH COMMUNICATION? — DISCOVERY & RECOVERY
--------------------------------------------------------------------------------
If you cannot connect to the sensor (unknown baudrate, factory reset, or macOS
CH340 clock jitter preventing reception at 921600 baud), run:

    python witrecovery.py

This script automatically:
  1. Scans all supported baudrates and reports the one the sensor is using.
  2. If the scan fails (e.g. 921600 baud on macOS where the CH340 USB adapter
     produces timing jitter), blindly commands the sensor to switch to 9600 baud
     — a rate that works reliably on all platforms.

After running, use witconfig.py to reconfigure and save your preferred settings to flash.

--------------------------------------------------------------------------------
3. GENERAL DRIVER ARCHITECTURE & POWER FEATURES
--------------------------------------------------------------------------------
To replace the complex, multi-file official SDK, this driver implements several
low-level features under a clean, thread-safe Python interface:

*   Auto-Discovery (Plug-and-Play):
    You do not need to guess or hunt for your port name (which varies by OS, e.g.,
    "COM3" on Windows, "/dev/ttyUSB0" on Linux, or "/dev/cu.usbserial-10" on macOS).
    Two module-level functions in witapi.py handle discovery automatically:
      - `discover_port()`: scans USB ports for the CH340 chip (VID 6790) and returns
        the port path.
      - `discover_baud()`: probes each supported baud rate and returns the active one.
    Pass the results directly to `WitMotionSensor(port=..., baud_bps=...)`.

*   Asynchronous Parsing Thread:
    When you call `stream()`, the driver flushes the OS serial input buffer and spawns
    a background thread that continuously reads and parses incoming bytes from the serial
    connection into WitMotion packets. The hardware streams range-scaled 16-bit integers
    in sensor-native units (g, dps); the parser converts all values to SI units
    (m/s², rad/s, µT, rad, °C) before writing the completed frame into a single-slot
    variable (`latest`), always overwriting the previous frame. Your main program
    retrieves the latest frame by calling `get_reading()`.

*   Connection Health & Baud Rate Validation:
    To prevent silent failures, right after opening the port, `connect()` queries
    the sensor's registers synchronously. If the sensor is non-responsive
    (e.g., if the baudrate is set incorrectly, or if you run into Mac USB timing issues
    at 921600 baud), it cleanly closes the port and raises a `ConnectionError` before
    any reading thread is started.

*   Queue-based Frame Dispatch (Trigger Logic):
    The sensor streams separate packets in a hardwired sequence:
      Time -> Accelerometer -> Gyroscope -> Euler Angles -> Magnetometer -> Quaternion
    Instead of pushing a partial frame for each packet, the driver inspects your enabled
    modalities, determines which packet is the *last* in the sequence, and writes a
    complete data frame exactly once per measurement cycle. This ensures the data you
    retrieve via `get_reading()` is always fully synchronized.

*   Software PLL (Clock Alignment & Skew Compensation):
    A Software Phase-Locked Loop (PLL) is used to calculate ultra-precise timestamps.
    It serves two key purposes:
      1. When hardware timestamps are enabled (via register 0x02): The PLL dynamically
         tracks and compensates for the physical clock drift (skew) between the sensor's
         microcontroller oscillator and the host PC clock. It synchronizes the two time
         scales, preventing timing drift in long-term recordings.
      2. When hardware timestamps are disabled (to save bandwidth): The PLL acts as a
         jitter filter. It estimates the true underlying sampling rate of the microcontroller
         and generates smooth timestamps, entirely erasing host OS scheduling latencies and
         USB packet bundling jitter.

--------------------------------------------------------------------------------
4. CONFIGURATION & CALIBRATION FEATURES
--------------------------------------------------------------------------------
Settings written via the API go to volatile registers by default and are lost on power
cycle. Call `save_config()` to persist them to flash. The key configuration and
calibration features are:

*   Baud Rate & Return Rate:
    You can configure the serial transfer speed (9600 to 921600 baud) and the streaming
    frequency (0.2 Hz to 1000 Hz). The API includes safety checks that block you from
    exceeding the bandwidth capacity of your serial connection.

*   Hardware Low-Pass Filter (LPF) Cutoff:
    Configures the physical analog anti-aliasing filter bandwidth on the sensor (5 Hz to 256 Hz)
    to prevent aliasing of high-frequency noise and vibrations into your signals.
    * General Guidance: According to the Nyquist theorem, to prevent spectral aliasing,
      the cutoff frequency should be configured to about 40% of your streaming frequency
      (e.g., set to 188 Hz for 500 Hz streaming, or 20 Hz for 50 Hz streaming).
    * HWT906 hardware limitation: The analog LPF circuit supports a minimum of 5 Hz.
      Register value 0x07 (2 Hz) exists in the generic WitMotion protocol docs and may
      work on other WitMotion sensors, but the HWT906 silently ignores it.

*   Manual Zero Bias Offsets:
    Allows reading and writing zero-bias calibration offset vectors for the Accelerometer,
    Gyroscope, and Magnetometer directly to hardware registers.

*   Accelerometer Self-Calibration:
    To align the sensor to gravity: place the sensor completely flat, level, and stationary,
    then trigger the calibration. The driver blocks for ~6 seconds while the sensor
    averages out zero-biases and aligns its Z-axis to 1g (9.80665 m/s²).

*   Magnetic Field Calibration:
    Used to eliminate local magnetic disturbances (e.g., from nearby metals). You start
    the calibration, slowly rotate the sensor in all three dimensions (drawing a sphere
    in the air), and then stop the calibration. The sensor calculates soft/hard-iron
    compensations and commits them to flash.

--------------------------------------------------------------------------------
5. PRIORITY API REFERENCE (MOST RELEVANT TO LEAST RELEVANT)
--------------------------------------------------------------------------------

A. Core Connection & Data Retrieval Methods
================================================================================
from witapi import WitMotionSensor, discover_port, discover_baud
port = discover_port()           # find the CH340 serial port
baud = discover_baud(port)       # detect the active baud rate
sensor = WitMotionSensor(port=port, baud_bps=baud)
    Initializes the driver.
    * port (str): Serial port path (e.g., "/dev/cu.usbserial-10").
    * baud_bps (int): Serial communication speed in bps. Must match the sensor's
      current baud rate. Use discover_baud() if unknown.

    Exposed Attributes (set after connect()):
    * sensor.rate (int): Streaming rate in Hz, read from hardware.
    * sensor.filt (int): LPF cutoff in Hz, read from hardware.
    * sensor.acc_range (float): Accelerometer measurement range in g (e.g. 16.0).
    * sensor.gyro_range (float): Gyroscope measurement range in dps (e.g. 2000.0).

sensor.connect()
    Auto-discovers the port (if port was None), opens the serial connection,
    and runs communication health checks synchronously. Also applies the configured
    rate, LPF, and disables gyro auto-cal on the hardware. Does NOT start the
    background parsing thread.
    * Raises ConnectionError if the sensor does not respond after 10 attempts.
      Run api/witrecovery.py if the baudrate is unknown or comm fails on macOS.

sensor.set_outputs(ts=True, acc=True, gyro=True, ang=False, mag=True, quat=False)
    Configures which sensor output packets are streamed. Must be called before
    stream(). Includes bandwidth checks to prevent serial saturation.
    Parameters: ts (hardware timestamp), acc (accelerometer), gyro (angular velocity),
    ang (Euler angles), mag (magnetometer), quat (quaternion).

sensor.stream()
    Flushes the OS serial input buffer and starts the background packet parsing thread.
    Returns immediately (non-blocking). Call disconnect() to stop.

sensor.get_reading()
    Returns the latest measurement frame (non-destructive), or None if no frame has
    been received yet. The slot always holds the most recent frame and is NOT cleared
    on read — repeated calls return the same frame until the parser overwrites it with
    a new one. Use the `pll_time_s` timestamp to detect new data:

        last_time = None
        data = sensor.get_reading()
        if data is not None and data["pll_time_s"] != last_time:
            last_time = data["pll_time_s"]
            # process new frame

    This non-destructive semantic is especially useful with multiple sensors: each
    sensor always has a valid frame available, so a slow sensor never blocks an ESKF
    update for the others.

sensor.get_data_header()
    Returns a CSV header string matching the currently configured output fields.
    Column order, suffixes, and SI units reflect exactly what each frame from
    get_reading() contains, based on the last call to set_outputs().
    Example: "pll_time_s [s], acc_x [m/s²], acc_y [m/s²], acc_z [m/s²], ..."
    Typical use: write as the first line of a log file before streaming data.

sensor.disconnect()
    Stops the background parsing thread (if running) and closes the serial port connection.

B. Hardware Configuration Methods
================================================================================
sensor.set_streaming_rate(rate_hz)
    Sets the sensor output rate (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 125, 200, 500, 1000 Hz).
    * Also updates the internal Software PLL parameters to match the new output rate.

sensor.set_lpf(cutoff_freq_hz)
    Configures the hardware low-pass filter cutoff frequency (5, 10, 20, 42, 98, 188, 256 Hz).

sensor.set_gyro_range(range_dps)
    Sets the gyroscope full-scale range (250, 500, 1000, 2000 dps).

sensor.set_acc_range(range_g)
    Sets the accelerometer full-scale range (2, 4, 8, 16 g).

sensor.set_baudrate(baud)
    Sets the serial communication baud rate. Requires reconnection to take effect.

sensor.set_gyro_autocal(enabled=False)
    Enables/disables the sensor's automatic zero-gyro calibration when static.
    * Note: Disabled by default to prevent sensor-level drift correction from fighting
      your custom orientation filter (like ESKF).

sensor.set_algorithm_axis(use_6axis=False)
    Selects the attitude fusion algorithm:
    * False (default): 9-axis — uses magnetometer to stabilize yaw.
    * True: 6-axis — yaw is computed relative to starting position (drifts over time).

sensor.set_filter_coefficient(filtk_val)
    Sets the FILTK smoothing coefficient for the internal attitude filter (1–10000).
    Larger values smooth orientation output at the cost of phase lag.

sensor.set_led(enabled=True)
    Enables or disables the status LED on the sensor module.

sensor.set_mounting_orientation(vertical=False)
    Configures horizontal (default) or vertical mounting layout.

sensor.show_config()
    Queries all main registers on the physical sensor and prints a complete state
    report (modalities, rate, baud, LPF, gyro/acc range, mounting, algorithm,
    FILTK, gyro auto-cal, bias offsets) to the console.

C. Hardware Calibration & Advanced Methods
================================================================================
sensor.calibrate_accelerometer()
    Starts gravity alignment calibration. Blocks the calling thread for ~6.0 seconds.
    * IMPORTANT: Sensor must be flat and still during this time.

sensor.start_magnetic_calibration()
    Places the sensor into magnetic calibration mode.
    * Slowly rotate the sensor in 3D space after calling this.

sensor.stop_magnetic_calibration()
    Saves the computed soft/hard-iron magnetic calibrations to flash memory and exits calibration.

sensor.get_offsets()
    Reads the current manual zero-bias offset registers for acc, gyro, and mag.
    Returns a dict with keys 'acc_offset', 'gyro_offset', 'mag_offset'.

sensor.set_offsets(ax, ay, az, gx, gy, gz, mx, my, mz)
    Writes custom bias offsets to hardware registers (signed 16-bit integers).

sensor.save_config()
    Saves all current register values to the sensor's flash memory.

sensor.reboot()
    Sends a soft reboot command to the sensor module.

sensor.factory_reset(restore_baud=True)
    Restores the sensor to factory defaults.
    * If restore_baud is True, automatically reconnects at 921600 baud and reconfigures
      the baud rate back to the operating value before the reset.

D. Low-Level Register Access (Encapsulated / Private)
================================================================================
To protect the user from sending commands that could bypass safety limits (like
the bandwidth capacity checks) or cause communication lockups, low-level register
access has been encapsulated behind private helper methods:

sensor._write_register(reg_addr, value)
    Writes a 16-bit integer value directly to a hardware register.

sensor._read_registers(start_reg, count=4)
    Queries a block of registers starting at start_reg. Returns a list of `count`
    integers (1 to 4). Raises TimeoutError if the sensor does not respond.

Most users should not use these low-level methods and should instead use the
high-level configuration methods listed above.

--------------------------------------------------------------------------------
6. DATA FRAME STRUCTURE & UNITS
--------------------------------------------------------------------------------
Each frame returned by `sensor.get_reading()` is a dictionary containing:

Frames are adaptive: only keys for enabled outputs are present. Disabled outputs
are absent from the dict entirely — no None values.

NOTE: The hardware streams range-scaled 16-bit integers in sensor-native units
(g for acceleration, dps for angular velocity). The driver converts all values
to SI units before returning them. The units listed below are driver output units.

*   `pll_time_s` (float): Software PLL smoothed timestamp in seconds. Always present.
    Maintains phase-lock with the hardware clock, filters out OS packet bundling jitter,
    and automatically compensates for drift between the microcontroller's oscillator
    and the host PC clock.
*   `time_s` (float): Hardware clock timestamp in seconds, elapsed since the first
    received packet. Present only when ts=True in set_outputs().
*   `acc` (list of 3 floats): Accelerometer [X, Y, Z] in **m/s²**
    (using _STANDARD_GRAVITY = 9.80665 m/s²). At rest on a flat surface, Z ≈ +9.81.
*   `gyro` (list of 3 floats): Angular velocity [X, Y, Z] in **rad/s**.
*   `mag` (list of 3 floats): Magnetometer [X, Y, Z] in **µT**.
*   `temp` (float): Sensor die temperature in **°C**. Present when acc is enabled.
*   `angle` (list of 3 floats): Fused Euler angles [roll, pitch, yaw] in **rad**.
    Computed onboard using the sensor's internal filter. Behavior is controlled by
    `set_algorithm_axis()` (9-axis vs 6-axis fusion) and `set_filter_coefficient()`
    (smoothing vs. latency tradeoff).
*   `quat` (list of 4 floats): Fused orientation as [q0, qx, qy, qz]
    (scalar first, matching hardware field order).

--------------------------------------------------------------------------------
7. COMPLETE PROGRAMMING EXAMPLE
--------------------------------------------------------------------------------
Save this code in a Python file and run it (ensuring pyserial is installed):

See api/witexample.py for a complete, runnable streaming example.

A minimal inline version:

```python
import sys, time
from witapi import WitMotionSensor, discover_port, discover_baud

sensor = None
try:
    port = discover_port()  # replace with port name if known
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
```
================================================================================
