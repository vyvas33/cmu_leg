# WitMotion HWT906 Python Driver

This README describes the architecture, configuration features, and programming interface for the custom, single-file Python driver `WitMotionSensor` located in `drivers/witmotion/hwt906.py`.

*hgeyer, Jun 2026*

---

## 1. Introduction: What is the HWT906 Sensor?

The WitMotion HWT906 is an Inertial Measurement Unit (IMU) and Attitude and Heading Reference System (AHRS). It contains:
* **Accelerometer**: Measures linear acceleration along 3 axes: X, Y, Z.
* **Gyroscope**: Measures angular rotation speed about these 3 axes: X-Roll, Y-Pitch, Z-Yaw.
* **Magnetometer**: Measures the surrounding magnetic field along these 3 axes (for heading).
* **Temperature Sensor**: Measures the sensor die temperature in °C. Reported alongside accelerometer data (present when acc output is enabled).
* **Onboard Processor**: Merges these raw signals to compute the IMU orientation (Euler angles or Quaternions) relative to the Earth, assuming an ENU navigation frame (Xn-East, Yn-North, Zn-Up)

This driver establishes a connection to the sensor over USB (via a virtual serial port), configures its hardware parameters, and streams IMU data directly to your code.

---

## 2. Cannot establish communication? — Discovery & Recovery

If you cannot connect to the sensor (unknown baudrate, factory reset), run:

```bash
python recovery.py
```

This script automatically:
1. Scans all supported baudrates and reports the one the sensor is using.
2. If the scan fails (e.g. 921600 baud on macOS where the CH340 USB adapter used by the sensor breakout board produces timing jitter), blindly commands the sensor to switch to 9600 baud — a rate that works reliably on all platforms.

After running, use `settings.py` to reconfigure and save your preferred settings to the flash memory of the sensor.

---

## 3. General Driver Architecture & Data Access

This driver replaces the official SDK and implements low-level features under a single, thread-safe Python interface:

### Auto-Discovery (Plug-and-Play)
You do not need to hunt for your port name. Two module-level functions in `hwt906.py` handle discovery:
* `discover_port()`: scans USB ports for the CH340 communication chip (VID 6790) and returns the port path.
* `discover_baud()`: probes each supported baud rate and returns the active one.

### Asynchronous Parsing Thread
When you call `stream()`, the driver spawns a background thread that reads and parses the data continuously sent by the sensor. The parser converts all values to SI units (m/s², rad/s, µT, rad, °C) and dispatches complete, synchronized data sets to both a real-time snapshot slot and a logging queue.

### Data Access Approaches
The driver provides two distinct ways to access parsed data sets, depending on your application:

1.  **Real-time Polling (for Control/Prediction)**:
    Access the `latest_data` property. It returns the most recently parsed set (Zero-Order Hold). This is non-destructive and ideal for low-latency control loops or real-time visualization.
2.  **Gapless Logging (for Trials/Post-processing)**:
    Call the `drain_data_buffer()` method. It returns a list of all sets received since the last call and clears the internal logging queue. This ensures that no intermediate samples are lost due to OS scheduling jitter.

### Host Timestamp Estimation (α-filter)
The driver uses a first-order α-filter (predictor-corrector) to produce a host-side timestamp (`host_time_s`). It is always the timestamp to use — it lives in the host PC's `perf_counter()` timebase, making it trivially aligned with any other data logged on the PC (camera frames, user input, etc.). Its quality, however, depends on whether the hardware time packet is enabled.

**Recommended configuration: always enable `ts=True` in `set_outputs()`.**

With the hardware timestamp enabled, the α-filter uses the inter-packet elapsed time from the sensor's own clock to drive the prediction step. This eliminates OS scheduling jitter and USB batch-delivery artifacts entirely, producing a smooth, accurate timeline suitable for demanding filters like the ESKF.

Without the hardware timestamp, the α-filter falls back to PC arrival time differences to estimate the prediction step. Because USB packets are frequently batched by the OS, these arrival times carry large pseudo-jitter that the filter cannot distinguish from real timing variation. The result is a usable but degraded estimate — adequate when bandwidth constraints force disabling `ts`, but not the preferred operating mode.

#### How it works:
Each time a complete data frame arrives, the driver reads `perf_counter()` and subtracts the session start time to get a raw elapsed measurement. The α-filter dead-reckons a prediction forward by `n` nominal sample periods (`n > 1` signals dropped packets), then nudges it toward the raw measurement with gain α = 0.02. The small gain rejects frame-to-frame noise while tracking long-term clock drift.

*The raw sensor hardware clock (`time_s`) is present only when `ts=True` and is used internally to drive the α-filter prediction. It is available for diagnostics or cross-validation but is not intended as the primary timestamp for users.*

---

## 4. Configuration & Calibration Features

Settings written via the driver change sensor registers without committing to the sensor's flash memory by default. Call `save_config()` to persist them to flash.

### Hardware Low-Pass Filter (LPF) Cutoff
Configures the physical analog anti-aliasing filter on the sensor (5 Hz to 256 Hz).
* **Guidance**: To prevent aliasing, the cutoff should be set to ~40% of your streaming frequency (e.g., set to 188 Hz for 500 Hz streaming).

### Accelerometer & Magnetic Calibration
* `calibrate_accelerometer()`: Aligns the sensor to gravity. Sensor must be flat and perfectly still for ~6 seconds.
* `start_magnetic_calibration()` / `stop_magnetic_calibration()`: Eliminates local magnetic disturbances. Rotate the sensor slowly in all 3D orientations while active.

---

## 5. Priority Driver Reference

### A. Core Connection & Data Retrieval
```python
from drivers.witmotion import WitMotionSensor

sensor = WitMotionSensor(port=YOUR_PORT_NAME, baud_bps=YOUR_SENSOR_BAUDRATE)
sensor.connect()
sensor.set_streaming_rate(YOUR_STREAMING_RATE)
sensor.set_lpf(YOUR_LOWPASS_CUTOFF_FREQ)
sensor.set_outputs(acc=True, gyro=True, mag=True)  # select which information to stream
sensor.stream()
...
data = sensor.latest_data  # for realtime applications
data = sensor.drain_data_buffer()  # for logging
...
sensor.stop_streaming()       # pause streaming; port stays open for reconfiguration
sensor.set_streaming_rate(500)  # reconfigure while connected
sensor.stream()               # resume
...
sensor.stop_streaming()
sensor.sleep()                # enter low-power mode (wakes automatically on next connect())
sensor.disconnect()           # stop streaming, sleep, and close the port
```

#### `sensor.connect()`
Opens the serial port and verifies communication. Syncs current hardware settings (rate, LPF, ranges) into the driver state.

#### `sensor.set_streaming_rate(rate_hz) [optional]`
Sets the sensor's data streaming rate (0.2 to 1000 Hz).

#### `sensor.set_lpf(cutoff_hz) [optional]`
Configures the hardware low-pass filter (5 to 256 Hz) to prevent aliasing.

#### `sensor.set_outputs(ts=True, acc=True, gyro=True, ang=False, mag=True, quat=False) [optional]`
Selects which data packets are streamed (Timestamp, Acc, Gyro, Euler, Mag, Quat). Includes bandwidth safety checks.

#### `sensor.stream()`
Starts the background parsing thread to begin receiving data.

#### `sensor.latest_data` (property)
Returns the most recently parsed data set (dict), or `None` if no data has been received yet. Non-destructive.

#### `sensor.drain_data_buffer()`
Returns a `list` of all data sets (dicts) received since the last call (max 5 seconds, independent of streaming rate) and clears the buffer. Ideal for gapless logging.

#### `sensor.get_data_header()`
Returns a CSV header string matching the currently configured output fields (e.g., `"host_time_s [s], acc_x [m/s²], ..."`).

#### `sensor.stop_streaming()`
Stops the background parsing thread without closing the serial port. Use this to pause streaming so you can reconfigure the sensor (e.g. change rate or outputs) and then call `stream()` again to resume. No-op if not currently streaming.

#### `sensor.sleep()`
Puts the sensor into low-power hibernation (register 0x22 = 0x0001). Requires `stop_streaming()` first (`@_config_only` guard). The sensor wakes on any UART activity — `connect()` handles this implicitly via the register read commands it sends. Sleep is not persisted to flash and resets to active on power cycle. Called automatically by `disconnect()`.

#### `sensor.disconnect()`
Stops the parsing thread (via `stop_streaming()`), sleeps the sensor, and closes the serial port.

### B. Hardware Configuration
* `set_streaming_rate(rate_hz)`: Sets output frequency (0.2 to 1000 Hz).
* `set_lpf(cutoff_hz)`: Configures hardware LPF (5 to 256 Hz).
* `set_acc_range(range_g)` / `set_gyro_range(range_dps)`: Sets full-scale measurement ranges.
* `set_algorithm_axis(use_6axis=False)`: Toggles between 9-axis (stabilized yaw) and 6-axis (relative yaw) fusion.
* `set_gyro_autocal(enabled=False)`: Toggles automatic zero-gyro calibration when still. Disable if you use your own sensor fusion (such as ESKF in PDR applications).
* `set_mounting_orientation(vertical=False)`: Configures horizontal or vertical mounting.
* `set_led(enabled=True)`: Toggles the status LED on the sensor module.

### C. System & Diagnostics
* `show_config()`: Prints a report of all hardware registers to the console.
* `save_config()`: Persists all current registers to the sensor's flash memory.
* `sleep()`: Enters low-power hibernation (register 0x22). Called automatically by `disconnect()`. Requires `stop_streaming()` first.
* `reboot()`: Restarts the sensor (takes about 2 seconds). Requires reconnect.
* `factory_reset()`: Resets the sensor to its factory defaults including baudrate, which will change to 921600 bits per second. Requires reconnect.

---

## 6. Data Set Structure & Units

Each set returned by `latest_data` or `drain_data_buffer()` is a dictionary. Only keys for enabled outputs (defined by `set_outputs()`) are present.

**Units**: The driver converts all values to SI units.

* `host_time_s` (float): **[s]** Primary timestamp (Recommended). α-filtered host-side elapsed time. Always present.
* `time_s` (float): **[s]** Hardware clock timestamp (Optional). Raw sensor elapsed time; present only when `ts=True`.
* `acc` (list of 3 floats): **[m/s²]** Linear acceleration [X, Y, Z].
* `gyro` (list of 3 floats): **[rad/s]** Angular velocity [X, Y, Z].
* `mag` (list of 3 floats): **[µT]** Magnetic field [X, Y, Z].
* `angle` (list of 3 floats): **[rad]** Euler angles [roll, pitch, yaw].
* `quat` (list of 4 floats): **[-]** Orientation as [q0, qx, qy, qz] (scalar first).
* `temp` (float): **[°C]** Sensor die temperature.

---

## 7. Programming Example

### Real-time Polling
```python
while True:
    data = sensor.latest_data
    if data:
        print(f"Current Acc Z: {data['acc'][2]:.2f} m/s²")
    time.sleep(0.01)
```

### Gapless Logging
```python
with open("log.csv", "w") as f:
    f.write(sensor.get_data_header() + "\n")
    while True:
        batch = sensor.drain_data_buffer()
        for frame in batch:
            f.write(",".join(map(str, frame.values())) + "\n")
        time.sleep(0.1)
```
