# coding:UTF-8
"""
WitMotion HWT906 Inclinometer Driver.

A self-contained driver that communicates directly with the WITMOTION HWT906
IMU/AHRS sensor over a serial (TTL) interface. It handles asynchronous binary
stream parsing, cycle-slip corrected Software PLL, lock-protected thread-safe
register queries, calibration controls, and single-slot latest-frame delivery.

Typical Usage:
    from witapi import WitMotionSensor, discover_port, discover_baud

    port = discover_port()
    baud = discover_baud(port)
    sensor = WitMotionSensor(port=port, baud_bps=baud)
    sensor.connect()

    sensor.set_outputs(ts=True, acc=True, gyro=True)
    sensor.stream()

    data = sensor.get_reading()
    if data:
        print(data["acc"], data["pll_time_s"])

    sensor.disconnect()

hgeyer, June 2026
"""

import functools
import math
import struct
import threading
import time

import serial
import serial.tools.list_ports

_CH340_VID = 6790  # vendor ID of the CH340 USB-to-UART bridge on the HWT906 eval board


def discover_port() -> str:
    """
    Return the serial port of the single connected CH340 USB-to-UART bridge.

    Raises:
        ValueError: Multiple CH340 devices found — specify port explicitly.
        FileNotFoundError: No CH340 device found.
    """
    ports = [p.device for p in serial.tools.list_ports.comports() if p.vid == _CH340_VID]
    if len(ports) == 1:
        return ports[0]
    if len(ports) > 1:
        raise ValueError(f"Multiple CH340 ports found: {ports}. Specify port explicitly.")
    raise FileNotFoundError("No CH340 device found. Check sensor connection and drivers.")


def discover_baud(port: str) -> int:
    """
    Probe each known baud rate and return the first one at which the sensor responds.

    Sends time-only / 50 Hz commands and counts 0x55 frame headers for 0.5 s per rate.

    Raises:
        ConnectionError: Sensor did not respond at any baud rate.
            Run witrecovery.py for blind recovery.
    """
    for baud in WitMotionSensor._BAUD_MAP:
        try:
            with serial.Serial(port, baud, timeout=0.05) as ser:
                for cmd in (
                    bytes([0xFF, 0xAA, 0x02, 0x01, 0x00]),
                    bytes([0xFF, 0xAA, 0x03, WitMotionSensor._RATE_MAP[50], 0x00]),
                ):
                    ser.write(WitMotionSensor._UNLOCK_CMD)
                    time.sleep(0.05)
                    ser.write(cmd)
                    time.sleep(0.05)
                count, deadline = 0, time.perf_counter() + 0.5
                while time.perf_counter() < deadline:
                    chunk = ser.read(ser.in_waiting or 1)
                    if chunk:
                        count += chunk.count(WitMotionSensor._FRAME_HEADER)
        except serial.SerialException:
            continue
        if count >= 10:
            return baud
    raise ConnectionError(f"No response on {port} at any baud rate. Run witrecovery.py for recovery.")


def _config_only(method):
    """
    Decorator that prevents modification of configuration while the sensor is streaming.
    Any method decorated with @_config_only will raise a RuntimeError if called while
    the sensor is streaming.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._listening:
            raise RuntimeError("Call disconnect() before changing configuration.")
        return method(self, *args, **kwargs)

    return wrapper


class WitMotionSensor:
    """
    Driver class for the WITMOTION HWT906 sensor.

    This driver replaces the entire multi-file WitMotion SDK, packaging all serial
    communication, register read/write protocols, and scaling logic into a single
    thread-safe module.

    SI Unit Boundary:
        The hardware streams range-scaled 16-bit integers. The driver converts all
        physical quantities to SI units before exposing them via get_reading():
          - Acceleration:    m/s²  (using _STANDARD_GRAVITY = 9.80665 m/s²)
          - Angular velocity: rad/s
          - Magnetic field:   µT
          - Euler angles:     rad
        Callers are not expected to apply any further unit conversion.

    Configuration Features & Hardware Registers:
    ===========================================================================
    1. System Configuration Commands (Register 0x00)
       - Save Settings: Write 0x0000 to save current configurations to flash.
       - Factory Reset: Write 0x0001 to restore all settings to default.
       - Soft Reboot: Write 0x00FF to reboot the sensor module.

    2. Calibration Controls (Register 0x01)
       - Accelerometer Self-Calibration: Write 0x0001 to start calibration.
         * CRITICAL: Sensor must be placed flat and completely still.
         * This blocks the calling thread for ~5.5s.
       - Magnetic Field Calibration:
         * Start: Write 0x0007. Rotate the sensor slowly in all 3D orientations.
         * Stop: Write 0x0000. Saves the computed magnetic calibration offsets to flash.

    3. Content Output (Return Status Word / Modalities) (Register 0x02)
       - RSW (Return Status Word) is a bitmask specifying which data packets are streamed:
         * Bit 0: Time packet (0x50)
         * Bit 1: Accelerometer/Temperature packet (0x51)
         * Bit 2: Angular Velocity packet (0x52)
         * Bit 3: Euler Angles packet (0x53)
         * Bit 4: Magnetometer packet (0x54)
         * Bit 9: Quaternion packet (0x59) — output as [w, x, y, z] (scalar first, matching hardware)
       - Note: The WitMotion hardware streams active packets in a hardwired sequence:
         0x50 (Time) < 0x51 (Acc) < 0x52 (Gyro) < 0x53 (Angle) < 0x54 (Mag) < 0x59 (Quat)
         The driver automatically identifies the highest packet ID in this sequence
         and pushes the completed frame to the data queue immediately after this
         packet is parsed, ensuring zero-latency data access.

    4. Output Return Rate (Register 0x03)
       - Target streaming frequency in Hz.
       - Supported values: 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 125, 200, 500, 1000 Hz.

    5. Serial Baud Rate (Register 0x04)
       - Communication baud rate. Re-connection is required if changed.
       - Supported values: 9600, 115200, 230400, 460800, 921600.

    6. Manual Zero Bias Offsets (Registers 0x05 to 0x0D)
       - Read/write bias offset vectors as signed 16-bit integers:
         * 0x05, 0x06, 0x07: Acceleration X, Y, Z offsets.
         * 0x08, 0x09, 0x0A: Gyroscope X, Y, Z offsets.
         * 0x0B, 0x0C, 0x0D: Magnetometer X, Y, Z offsets.

    7. LED Control (Register 0x1B)
       - Enable/disable status LED.
       - Range: 0 = LED Enabled (On), 1 = LED Disabled (Off).

    8. Hardware Low-Pass Filter (LPF) Cutoff Frequency (Register 0x1F)
       - Configures the sensor's physical anti-aliasing low-pass filter.
       - Supported cutoff frequencies: 5, 10, 20, 42, 98, 188, 256 Hz.
       - Note: This affects raw measurements (Acc/Gyro). For a frame rate 'rate', the LPF
         cutoff frequency should be set to about 0.4 * rate to avoid signal pollution by aliasing.
       - HWT906 hardware limitation: The analog LPF circuit on the HWT906 supports a minimum
         cutoff of 5 Hz (per the datasheet bandwidth spec of 5~256 Hz). Register value 0x07
         (2 Hz) appears in the generic WitMotion protocol documentation and may be valid on
         other WitMotion sensors, but the HWT906 silently ignores it.

    9. Gyroscope Measurement Range (Register 0x20)
       - Maximum angular velocity range.
       - Supported values: 250, 500, 1000, 2000 dps.

    10. Accelerometer Measurement Range (Register 0x21)
        - Maximum acceleration range.
        - Supported values: 2, 4, 8, 16 g.

    11. Mounting Orientation (Register 0x23)
        - Mount layout.
        - Range: 0 = Horizontal mounting (default), 1 = Vertical mounting.

    12. Attitude Algorithm Axis Mode (Register 0x24)
        - Axis mode for internal orientation algorithm.
        - Range: 0 = 9-axis (uses magnetometer to stabilize yaw).
                 1 = 6-axis (yaw is calculated relative to starting position, drifts over time).

    13. Smoothing Filter Coefficient (FILTK) (Register 0x25)
        - Unitless weighting gain coefficient for the internal attitude output filter
          (complementary/Kalman filter state smoothing). The default value is 50.
        - Range: 1 (minimal smoothing, low latency, more noise) to 10000 (maximum smoothing, high latency, less noise).
        - Note: This affects only the attitude algorithm output (Euler/Quaternions), not raw Acc/Gyro.

    14. Gyroscope Auto-Calibration Control (Register 0x63)
        - Toggles automatic zero-gyro calibration on startup or when still.
        - Range: 0 = Enabled, 1 = Disabled.
        - Note: Gyro auto-calibration should be disabled (default) when running PDR algorithms
          with custom ESKF bias corrections, to prevent conflicts.
    ===========================================================================

    Attributes:
        port (str): Path to the serial port (e.g. '/dev/ttyUSB0' or '/dev/cu.usbserial-10').
        baud (int): Baud rate for communication.
        rate (float): Output streaming return rate in Hz. Set by connect().
        filt (int): Hardware LPF cutoff frequency in Hz. Set by connect().
        acc_range (float): Accelerometer measurement range in g. Set by connect().
        gyro_range (float): Gyroscope measurement range in dps. Set by connect().
    """

    # -------------------
    # Register value maps
    # -------------------

    # fmt: off
    _BAUD_MAP = {9600: 0x02, 115200: 0x06, 230400: 0x07, 460800: 0x08, 921600: 0x09}
    _RATE_MAP = {
        0.2: 0x01, 0.5: 0x02, 1: 0x03, 2: 0x04, 5: 0x05, 10: 0x06, 20: 0x07,
        50: 0x08, 100: 0x09, 125: 0x0A, 200: 0x0B, 500: 0x0C, 1000: 0x0D
    }
    _LPF_MAP = {5: 0x06, 10: 0x05, 20: 0x04, 42: 0x03, 98: 0x02, 188: 0x01, 256: 0x00}

    _GYRO_RANGE_MAP = {250: 0, 500: 1, 1000: 2, 2000: 3}
    _ACC_RANGE_MAP = {2: 0, 4: 1, 8: 2, 16: 3}
    _STANDARD_GRAVITY = 9.80665  # m/s² — SI standard gravity

    _FRAME_HEADER = 0x55
    _UNLOCK_CMD = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
    _OUTPUT_LIST_FIELDS = {
        0x51: ("acc", 3), 0x52: ("gyro", 3), 0x53: ("angle", 3), 0x54: ("mag", 3), 0x59: ("quat", 4)
    }
    # fmt: on

    def __init__(self, port: str = None, baud_bps: int = 921600):
        """
        Initialize the sensor driver.

        Args:
            port (str, optional): Serial port filename. If None, the driver will automatically
                discover the port by scanning for a CH340 USB-to-UART bridge (VID 6790).
            baud_bps (int): Communication baud rate in bits/s. Must match the sensor's current
                baud rate setting. Driver default: 921600 (factory hardware default).
        """
        # --- Serial connection ---
        self.port = port
        self.baud = baud_bps
        self._connection = None

        # --- Sensor hardware settings (assigned by connect()) ---
        self.rate: float
        self.filt: int
        self.acc_range: float
        self.gyro_range: float

        # --- Parser thread & data delivery ---
        self._parser_thread = None
        self._queue_lock = threading.Lock()
        self._listening = False

        self._packet_ids = set()  # packet IDs enabled via set_outputs()
        self._trigger_packet_id = None  # highest active ID; parser dispatches on its arrival
        self._acc_ndigits = None  # round() precision for acc output; derived from range LSB
        self._gyro_ndigits = None  # round() precision for gyro output; derived from range LSB
        self._sensor_data: dict = {  # in-place accumulation buffer; one slot per output
            "time_s": None,
            "pll_time_s": None,
            "acc": None,
            "gyro": None,
            "mag": None,
            "temp": None,
            "angle": None,
            "quat": None,
        }
        self._latest = None  # latest sensor data set accessed by get_reading()

        # --- Software PLL (None until stream() runs) ---
        self._pll_alpha = 0.02
        self._pll_sample_period: float | None
        self._pll_t_est: float | None
        self._start_pc_time: float | None
        self._start_sensor_ms: float | None
        self._sensor_time_ms: float | None

    # --------------------
    # 1. CONNECT TO SENSOR
    # --------------------

    def connect(self):
        """
        Open the serial port, verify communication, and sync hardware state.

        Auto-discovers the port if none was provided. Queries registers to confirm
        the sensor is responding at the configured baudrate, then reads the current
        hardware values for rate, LPF, and measurement ranges into driver state.
        Does NOT write any registers or start the background parsing thread.

        Raises:
            ConnectionError: Sensor did not respond after 10 verification attempts.
                If connection fails on macOS at high baud, run api/witrecovery.py
                to auto-discover the baudrate or recover communication.
        """
        # Auto-discover port if not explicitly provided
        if self.port is None:
            self.port = discover_port()
            print(f"WitMotion sensor discovered on port: {self.port}")

        print(f"Connecting to WitMotion sensor on {self.port} with baudrate={self.baud} bps: ", end="")
        self._connection = serial.Serial(self.port, self.baud, timeout=0.05)

        # Verify serial communication is working at the requested baudrate.
        # Retry up to 10 times to account for initial packet sync phase.
        verified = False
        for attempt in range(1, 11):
            try:
                self._read_registers(0x03, 1)
                verified = True
                break
            except TimeoutError:
                print(f"Baudrate verification attempt {attempt}/10 timed out...")
                time.sleep(0.1)

        if not verified:
            self.disconnect()
            print("\n" + "=" * 60)
            print("TROUBLESHOOTING BAUD RATE CONNECTION ERROR:")
            print("1. Check the physical USB connection and sensor power.")
            print("2. Ensure baud_bps matches the sensor's current setting.")
            print("3. If the baudrate is unknown or comm fails on macOS, run:")
            print("   python api/witrecovery.py")
            print("=" * 60 + "\n")
            raise ConnectionError(
                f"Communication verification failed at {self.baud} baud on port {self.port}. "
                "The sensor did not respond to register queries after 10 attempts."
            )

        print("success")

        # Reset parser state and timing variables
        self._listening = False
        self._start_sensor_ms = None
        self._sensor_time_ms = None
        self._pll_t_est = None
        self._start_pc_time = None

        # Sync sensor hardware state into driver: rate, LPF and ranges.
        rate_reg = self._read_registers(0x03, 1)
        self.rate = {v: k for k, v in self._RATE_MAP.items()}[rate_reg[0]]
        self._pll_sample_period = 1.0 / self.rate

        lpf_reg = self._read_registers(0x1F, 1)
        self.filt = {v: k for k, v in self._LPF_MAP.items()}[lpf_reg[0]]

        range_regs = self._read_registers(0x20, 2)
        self.gyro_range = float({v: k for k, v in self._GYRO_RANGE_MAP.items()}[range_regs[0]])
        self.acc_range = float({v: k for k, v in self._ACC_RANGE_MAP.items()}[range_regs[1]])
        self._acc_ndigits = math.ceil(-math.log10(self.acc_range / 32768.0 * self._STANDARD_GRAVITY))
        self._gyro_ndigits = math.ceil(-math.log10(self.gyro_range / 32768.0 * (math.pi / 180.0)))

    def disconnect(self):
        """
        Stop the reading thread (if running) and close the serial port.
        """
        if self._listening:
            self._listening = False
            self._parser_thread.join(timeout=1.0)
            self._parser_thread = None
            with self._queue_lock:
                self._latest = None
            print("Sensor streaming thread stopped.")

        if self._connection and self._connection.is_open:
            self._connection.close()
            print("WitMotion sensor disconnected.")

    # ----------------------
    # 2. COLLECT SENSOR DATA
    # ----------------------

    def stream(self):
        """
        Start the background packet parsing thread.

        Returns immediately (non-blocking). Use get_reading() to retrieve data.
        Call disconnect() to stop.
        """
        if not self._connection or not self._connection.is_open:
            raise RuntimeError("Sensor is disconnected. Call connect() first.")

        if self._listening:
            raise RuntimeError("Already streaming.")

        if not self._packet_ids:
            raise RuntimeError(
                "No outputs enabled. Call set_outputs() before stream() to select which data streams to receive."
            )

        with self._queue_lock:
            self._latest = None
        for k in self._sensor_data:
            self._sensor_data[k] = None
        for pid, (field, n) in self._OUTPUT_LIST_FIELDS.items():
            if pid in self._packet_ids:
                self._sensor_data[field] = [0.0] * n

        self._listening = True
        self._parser_thread = threading.Thread(
            target=self._host_side_parsing_thread, name="WitMotion-Parsing-Thread", daemon=True
        )
        self._parser_thread.start()
        print("Sensor streaming thread started.")

    def get_reading(self):
        """
        Return the latest parsed data set (non-destructive).

        Returns the same set on repeated calls until the parser overwrites it with
        a new one. Use the `pll_time_s` field to detect whether the set is new:

            timestamp = None
            data = sensor.get_reading()
            if data is not None and data["pll_time_s"] != timestamp:
                timestamp = data["pll_time_s"]
                # process new frame

        Returns:
            dict: The most recent data set, or None if no set has been received yet.
        """
        with self._queue_lock:
            return self._latest

    def get_data_header(self):
        """
        Return a CSV header string matching the currently configured output fields.

        The column order, suffixes, and units reflect exactly what each data set
        returned by get_reading() contains, based on the last call to set_outputs().

        Returns:
            str: Comma-separated column names with units, e.g.
                 "pll_time_s [s], acc_x [m/s²], acc_y [m/s²], acc_z [m/s²], ..."
        """
        active_fields = {"pll_time_s"}
        for pid, fields in [
            (0x50, ["time_s"]),
            (0x51, ["acc", "temp"]),
            (0x52, ["gyro"]),
            (0x53, ["angle"]),
            (0x54, ["mag"]),
            (0x59, ["quat"]),
        ]:
            if pid in self._packet_ids:
                active_fields.update(fields)

        columns = []
        for field, suffixes, unit in [
            ("time_s", [""], "s"),
            ("pll_time_s", [""], "s"),
            ("acc", ["_x", "_y", "_z"], "m/s²"),
            ("gyro", ["_x", "_y", "_z"], "rad/s"),
            ("mag", ["_x", "_y", "_z"], "µT"),
            ("temp", [""], "°C"),
            ("angle", ["_roll", "_pitch", "_yaw"], "rad"),
            ("quat", ["_0", "_x", "_y", "_z"], "-"),
        ]:
            if field in active_fields:
                columns.extend(f"{field}{s} [{unit}]" for s in suffixes)
        return ", ".join(columns)

    # -------------------
    # 3. CONFIGURE SENSOR
    # -------------------

    @_config_only
    def set_baudrate(self, baud):
        """
        Configure the communication baud rate of the sensor.

        Args:
            baud (int): Communication baud rate. Supported rates:
                9600, 115200, 230400, 460800, 921600.
        """
        if baud not in self._BAUD_MAP:
            raise ValueError(f"Unsupported baud rate: {baud}. Choose from {list(self._BAUD_MAP.keys())}")

        # Check bandwidth capacity
        num_active = len(self._packet_ids)
        self._check_bandwidth(num_active, self.rate, baud)

        self._write_register(0x04, self._BAUD_MAP[baud])
        self.baud = baud
        print(f"Sensor communication rate configured to {baud} baud. Re-connect required if changed.")

    @_config_only
    def set_outputs(self, ts=True, acc=True, gyro=True, ang=False, mag=True, quat=False):
        """
        Configure which sensor outputs are streamed (RSW register 0x02 bitmask).

        Checks whether the requested outputs will exceed the serial line bandwidth, and
        warns if it exceeds 50% or raises ValueError if it exceeds 75% of capacity.

        Formula:
            usage = 11 bytes/packet * num_outputs * rate (kB/s)
            capacity = baudrate / 10 bits/byte (kB/s)

        Note: The WitMotion hardware streams active packets in a hardwired sequence:
            0x50 (ts) < 0x51 (acc) < 0x52 (gyro) < 0x53 (ang) < 0x54 (mag) < 0x59 (quat)
        The driver identifies the highest active packet ID and pushes a complete
        frame to the data queue immediately after it is parsed.

        Args:
            ts (bool): Enable hardware timestamp packet (0x50).
            acc (bool): Enable Accelerometer/Temperature packet (0x51).
            gyro (bool): Enable Angular Velocity packet (0x52).
            ang (bool): Enable Euler Angles packet (0x53).
            mag (bool): Enable Magnetic Field packet (0x54).
            quat (bool): Enable Quaternion packet (0x59).
        """
        self._check_bandwidth(sum((ts, acc, gyro, ang, mag, quat)), self.rate, self.baud)

        val, active = 0, set()
        for enabled, bit, pid in [
            (ts, 0, 0x50),
            (acc, 1, 0x51),
            (gyro, 2, 0x52),
            (ang, 3, 0x53),
            (mag, 4, 0x54),
            (quat, 9, 0x59),
        ]:
            if enabled:
                val |= 1 << bit
                active.add(pid)

        if not active:
            raise ValueError("At least one output must remain enabled.")

        # Check current value before writing
        current_regs = self._read_registers(0x02, 1)
        if current_regs[0] != val:
            self._write_register(0x02, val)

        # Configure local trigger packet to match the last packet in transmission order
        self._packet_ids = active
        self._trigger_packet_id = max(active)

    @_config_only
    def set_streaming_rate(self, rate_hz):
        """
        Set the return rate (frequency) of data streaming.

        Args:
            rate_hz (float/int): Target frequency. Supported rates:
                0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 125, 200, 500, 1000 Hz.
        """
        if rate_hz not in self._RATE_MAP:
            raise ValueError(f"Unsupported rate: {rate_hz} Hz. Choose from {list(self._RATE_MAP.keys())}")

        # Check bandwidth capacity
        self._check_bandwidth(len(self._packet_ids), rate_hz, self.baud)

        target_val = self._RATE_MAP[rate_hz]
        current_regs = self._read_registers(0x03, 1)
        if current_regs[0] != target_val:
            self._write_register(0x03, target_val)
        self.rate = rate_hz
        self._pll_sample_period = 1.0 / rate_hz

    @_config_only
    def set_lpf(self, cutoff_freq_hz):
        """
        Set the Low-Pass Filter (LPF) bandwidth (Register 0x1F) for anti-aliasing.

        Args:
            cutoff_freq_hz (int): Low-pass filter cutoff frequency. Supported values:
                256, 188, 98, 42, 20, 10, 5 Hz.
        """
        if cutoff_freq_hz not in self._LPF_MAP:
            raise ValueError(f"Unsupported bandwidth: {cutoff_freq_hz} Hz. Choose from {list(self._LPF_MAP.keys())}")

        target_val = self._LPF_MAP[cutoff_freq_hz]
        current_regs = self._read_registers(0x1F, 1)
        if current_regs[0] != target_val:
            self._write_register(0x1F, target_val)
        self.filt = cutoff_freq_hz

    @_config_only
    def calibrate_accelerometer(self):
        """
        Execute accelerometer calibration.

        CRITICAL: This method blocks the calling thread for approximately 5.5 seconds
        (takes about 6 seconds total) while the sensor performs self-calibration. Keep the
        sensor flat and perfectly still.
        """
        print("Starting accelerometer calibration (takes about 6 seconds)... Keep the sensor flat and completely still.")
        self._write_register(0x01, 0x0001)
        time.sleep(5.5)  # Required hardware calibration settling time
        print("Accelerometer calibration complete.")

    @_config_only
    def start_magnetic_calibration(self):
        """
        Enter magnetic field calibration mode.
        Rotate the sensor slowly in 3D (around X, Y, and Z axes) while active.
        """
        print("Starting magnetic field calibration... Rotate the sensor slowly in all 3D directions.")
        self._write_register(0x01, 0x0007)

    @_config_only
    def stop_magnetic_calibration(self):
        """
        Exit magnetic field calibration.
        """
        print("Stopping magnetic field calibration...")
        self._write_register(0x01, 0x0000)
        time.sleep(0.1)
        print("Magnetic calibration complete. Call save_config() to persist offsets.")

    @_config_only
    def set_gyro_autocal(self, enabled=False):
        """
        Configure the automatic zero-gyro calibration on start/stillness (Register 0x63).

        Note: Gyro auto-calibration should be disabled (default) when running PDR algorithms
        with custom ESKF bias corrections, to prevent the hardware calibration from interfering
        with the software-estimated state.

        Args:
            enabled (bool): Enable (True) or disable (False) auto-gyro calibration (default: False).
        """
        val = 0 if enabled else 1
        current_regs = self._read_registers(0x63, 1)
        if current_regs[0] != val:
            self._write_register(0x63, val)

    @_config_only
    def set_gyro_range(self, range_dps):
        """
        Configure the measurement range of the gyroscope (Register 0x20).

        Args:
            range_dps (int): Max angular velocity range. Supported:
                250, 500, 1000, 2000 dps.
        """
        if range_dps not in self._GYRO_RANGE_MAP:
            raise ValueError(
                f"Unsupported gyroscope range: {range_dps} dps. Choose from {list(self._GYRO_RANGE_MAP.keys())}"
            )

        target_val = self._GYRO_RANGE_MAP[range_dps]
        current_regs = self._read_registers(0x20, 1)
        if current_regs[0] != target_val:
            self._write_register(0x20, target_val)
        self.gyro_range = float(range_dps)
        self._gyro_ndigits = math.ceil(-math.log10(self.gyro_range / 32768.0 * (math.pi / 180.0)))

    @_config_only
    def set_acc_range(self, range_g):
        """
        Configure the measurement range of the accelerometer (Register 0x21).

        Args:
            range_g (int): Max acceleration range. Supported:
                2, 4, 8, 16 g.
        """
        if range_g not in self._ACC_RANGE_MAP:
            raise ValueError(f"Unsupported accelerometer range: {range_g} g. Choose from {list(self._ACC_RANGE_MAP.keys())}")

        target_val = self._ACC_RANGE_MAP[range_g]
        current_regs = self._read_registers(0x21, 1)
        if current_regs[0] != target_val:
            self._write_register(0x21, target_val)
        self.acc_range = float(range_g)
        self._acc_ndigits = math.ceil(-math.log10(self.acc_range / 32768.0 * self._STANDARD_GRAVITY))

    @_config_only
    def set_led(self, enabled=True):
        """
        Configure the status LED on the sensor module.

        Args:
            enabled (bool): Turn LED on (True) or off (False).
        """
        val = 0 if enabled else 1
        current_regs = self._read_registers(0x1B, 1)
        if current_regs[0] != val:
            self._write_register(0x1B, val)

    @_config_only
    def set_mounting_orientation(self, vertical=False):
        """
        Configure the mounting orientation layout (Register 0x23).

        Args:
            vertical (bool): Set to False for horizontal mounting (default),
                or True for vertical mounting.
        """
        val = 1 if vertical else 0
        current_regs = self._read_registers(0x23, 1)
        if current_regs[0] != val:
            self._write_register(0x23, val)

    @_config_only
    def set_algorithm_axis(self, use_6axis=False):
        """
        Configure the attitude algorithm axis mode (Register 0x24).

        Args:
            use_6axis (bool): Set to False for 9-axis (uses magnetometer for heading),
                or True for 6-axis (yaw is calculated relative to starting position).
        """
        val = 1 if use_6axis else 0
        current_regs = self._read_registers(0x24, 1)
        if current_regs[0] != val:
            self._write_register(0x24, val)

    @_config_only
    def set_filter_coefficient(self, filtk_val):
        """
        Configure the sensor's internal attitude smoothing filter (Register 0x25).

        Note: This parameter controls the internal Kalman filtering for the calculated
        attitude (Euler angles/quaternions) only. It does NOT apply low-pass filtering to
        raw outputs (Acc/Gyro); raw filtering must be set using set_lpf().

        Args:
            filtk_val (int): Filter coefficient between 1 (minimal smoothing) and 10000 (maximum smoothing).
        """
        if not (1 <= filtk_val <= 10000):
            raise ValueError("Filter coefficient must be between 1 and 10000.")
        current_regs = self._read_registers(0x25, 1)
        if current_regs[0] != filtk_val:
            self._write_register(0x25, filtk_val)

    @_config_only
    def set_offsets(self, ax, ay, az, gx, gy, gz, mx, my, mz):
        """
        Write custom manual zero bias offsets to registers (0x05 to 0x0D).

        Args:
            ax, ay, az (int): Acceleration offset values.
            gx, gy, gz (int): Gyroscope offset values.
            mx, my, mz (int): Magnetometer offset values.
        """
        offsets = [ax, ay, az, gx, gy, gz, mx, my, mz]
        for i, val in enumerate(offsets):
            reg = 0x05 + i
            # Clamp and convert to unsigned 16-bit integer for registry writing
            val_u16 = val & 0xFFFF
            self._write_register(reg, val_u16)
            time.sleep(0.01)
        print("Custom manual bias offsets written to registers.")

    @_config_only
    def get_offsets(self):
        """
        Read the manual zero bias offset registers (0x05 to 0x0D).

        Returns:
            dict: Dictionary containing acceleration, gyro, and magnetometer bias vectors.
        """
        # Read blocks of registers starting at 0x05, 0x09, 0x0D
        block1 = self._read_registers(0x05, 4)
        time.sleep(0.02)
        block2 = self._read_registers(0x09, 4)
        time.sleep(0.02)
        block3 = self._read_registers(0x0D, 1)

        # Reinterpret the unsigned 16-bit register values as signed 16-bit integers
        ax, ay, az, gx = struct.unpack("<hhhh", struct.pack("<HHHH", *block1))
        gy, gz, mx, my = struct.unpack("<hhhh", struct.pack("<HHHH", *block2))
        mz = struct.unpack("<h", struct.pack("<H", block3[0]))[0]

        return {"acc_offset": [ax, ay, az], "gyro_offset": [gx, gy, gz], "mag_offset": [mx, my, mz]}

    @_config_only
    def save_config(self):
        """
        Save the current configuration registers to the sensor's flash memory.
        """
        self._write_register(0x00, 0x0000)
        time.sleep(0.6)  # flash write takes ~500ms; block until complete
        print("Sensor settings saved to flash.")

    @_config_only
    def reboot(self):
        """
        Perform a soft reboot of the sensor module.
        """
        self._write_register(0x00, 0x00FF)
        print("Sensor soft reboot command sent.")

    @_config_only
    def factory_reset(self, restore_baud=True):
        """
        Restore the sensor to its default factory configurations (Register 0x00).

        Note: A hardware factory reset reverts the communication baud rate to its
        default of 921600 bps. On macOS, the CH340 USB-to-UART clock jitter makes
        reading data at 921600 bps highly unreliable.

        Args:
            restore_baud (bool): If True (default), the driver will automatically
                reconnect blindly at 921600 bps after the reset, restore the baud rate
                back to the current operating baud rate (e.g., 460800), save it to flash,
                and reconnect the API. This ensures the user does not lose connection
                on Mac OS due to serial clock jitter.
        """
        current_baud = self.baud

        # Unlock and send factory reset command
        self._write_register(0x00, 0x0001)
        print("Sensor factory reset command sent.")

        # Disconnect immediately as the sensor is resetting and rebooting
        self.disconnect()
        time.sleep(3.0)  # Wait 3 seconds for sensor reboot/reset to finish
        print("Sensor reset to factory settings and disconnected.")

        if not restore_baud:
            self.baud = 921600
            return

        print(f"Restoring baud rate back to {current_baud} bps...")

        ser = serial.Serial(self.port, 921600, timeout=0.5)  # raw serial instance at factory-default baud
        time.sleep(3.0)  # macOS boot delay

        baud_val = self._BAUD_MAP[current_baud]

        # Send unlock + set baud -> unlock + save
        ser.write(self._UNLOCK_CMD)
        time.sleep(0.05)
        ser.write(bytearray([0xFF, 0xAA, 0x04, baud_val, 0x00]))
        time.sleep(0.05)

        ser.write(self._UNLOCK_CMD)
        time.sleep(0.05)
        ser.write(bytearray([0xFF, 0xAA, 0x00, 0x00, 0x00]))
        time.sleep(0.1)

        ser.close()
        print("Baud rate successfully restored.")

    def show_config(self):
        """
        Read all configuration registers from the sensor hardware and print a
        detailed state report.
        """

        if not self._connection or not self._connection.is_open:
            print("\n[Status] Sensor is currently DISCONNECTED. Cannot read hardware registers.")
            print("=" * 50 + "\n")
            return

        if self._listening:
            print("\n[Status] Sensor is actively streaming. Skipping hardware register queries to avoid collisions.")
            print("To read the hardware registers, call disconnect() first.")
            print("=" * 50 + "\n")
            return

        # Querying hardware registers
        chunk1 = self._read_registers(0x02, 4)
        led_reg = self._read_registers(0x1B, 1)
        chunk2 = self._read_registers(0x1F, 4)
        chunk3 = self._read_registers(0x23, 4)
        chunk4 = self._read_registers(0x63, 4)
        offsets = self.get_offsets()

        rsw = chunk1[0]
        enabled_outputs = [
            f"{lbl} ({pid})"
            for bit, lbl, pid in [
                (0, "Time", "0x50"),
                (1, "Acc", "0x51"),
                (2, "Gyro", "0x52"),
                (3, "Angle", "0x53"),
                (4, "Mag", "0x54"),
                (9, "Quat", "0x59"),
            ]
            if rsw & (1 << bit)
        ]
        print(f"Active Modalities (RSW 0x02):   {', '.join(enabled_outputs)}")
        rate_inv = {v: k for k, v in self._RATE_MAP.items()}
        baud_inv = {v: k for k, v in self._BAUD_MAP.items()}
        lpf_inv = {v: k for k, v in self._LPF_MAP.items()}
        gyro_inv = {v: k for k, v in self._GYRO_RANGE_MAP.items()}
        acc_inv = {v: k for k, v in self._ACC_RANGE_MAP.items()}

        rate_opts = ", ".join(str(k) for k in self._RATE_MAP)
        baud_opts = ", ".join(str(k) for k in self._BAUD_MAP)
        lpf_opts = ", ".join(str(k) for k in self._LPF_MAP)
        gyro_opts = ", ".join(str(k) for k in self._GYRO_RANGE_MAP)
        acc_opts = ", ".join(str(k) for k in self._ACC_RANGE_MAP)

        print(f"Output Return Rate (0x03):      {rate_inv[chunk1[1]]} Hz  [options: {rate_opts} Hz]")
        print(f"Baud Rate (0x04):               {baud_inv[chunk1[2]]} bps  [options: {baud_opts} bps]")
        print(f"LPF Cutoff Frequency (0x1F):    {lpf_inv[chunk2[0]]} Hz  [options: {lpf_opts} Hz]")

        hw_gyro_range = gyro_inv[chunk2[1]]
        gyro_res = hw_gyro_range / 32768.0
        print(
            f"Gyroscope Range (0x20):         ±{hw_gyro_range} dps with {gyro_res:.6f} dps/LSB [options: ±{gyro_opts} dps]"
        )

        hw_acc_range = acc_inv[chunk2[2]]
        acc_res = hw_acc_range / 32768.0
        print(
            f"Accelerometer Range (0x21):     ±{hw_acc_range} g with {acc_res:.6f} g/LSB and g = {self._STANDARD_GRAVITY} m/s² [options: ±{acc_opts} g]"
        )

        print(f"Mounting Orientation (0x23):    {'Vertical' if chunk3[0] == 1 else 'Horizontal'}")
        print(f"Attitude Algorithm (0x24):      {'6-axis' if chunk3[1] == 1 else '9-axis'}")
        print(f"Smoothing Filter FILTK (0x25):  {chunk3[2]}")

        print(f"Gyro Auto-Calibration (0x63):   {'Disabled' if chunk4[0] == 1 else 'Enabled'}")

        print("Manual Bias Offsets (0x05–0x0D)")
        print(f" ... Acc Offsets  (X,Y,Z):      {offsets['acc_offset']}")
        print(f" ... Gyro Offsets (X,Y,Z):      {offsets['gyro_offset']}")
        print(f" ... Mag Offsets  (X,Y,Z):      {offsets['mag_offset']}")

        print(f"Status LED (0x1B):              {'Disabled' if led_reg[0] == 1 else 'Enabled'}\n\n")

    # --------------------------------------
    # 4. LOW-LEVEL REGISTER ACCESS (PRIVATE)
    # --------------------------------------

    def _write_register(self, reg_addr, value):
        """
        Write a 16-bit value to a specific sensor configuration register.
        Automatically handles configuration unlocking.

        Args:
            reg_addr (int): Register address (e.g. 0x03 for Return Rate).
            value (int): 16-bit configuration value.
        """
        # Send unlock command (required before any register write)
        self._connection.write(self._UNLOCK_CMD)
        time.sleep(0.05)

        # Send write register command: 0xFF 0xAA [REG] [VAL_L] [VAL_H]
        write_cmd = bytearray([0xFF, 0xAA, reg_addr & 0xFF, value & 0xFF, (value >> 8) & 0xFF])
        self._connection.write(write_cmd)
        time.sleep(0.05)

    def _read_registers(self, start_reg, count=4):
        """
        Read a block of registers starting from a specific address.
        Performs a synchronous write-and-read directly on self._connection.

        Args:
            start_reg (int): Address of the first register to read.
            count (int): Number of registers to read (up to 4).

        Returns:
            list: List of count integer values.

        Raises:
            TimeoutError: If no register response is received within 50ms.
        """
        # Send read register query command: 0xFF 0xAA 0x27 [REG_L] [REG_H]
        read_cmd = bytearray([0xFF, 0xAA, 0x27, start_reg & 0xFF, (start_reg >> 8) & 0xFF])

        self._connection.write(read_cmd)

        buffer = bytearray()
        deadline = time.perf_counter() + 0.05
        while True:
            chunk = self._connection.read(max(1, self._connection.in_waiting))
            if not chunk or time.perf_counter() > deadline:
                raise TimeoutError(f"Register read query for address {start_reg:#04x} timed out.")
            buffer.extend(chunk)
            while True:
                result = self._next_packet(buffer)
                if result is None:
                    break
                ptype, payload = result
                if ptype == 0x5F:  # Register response
                    val0, val1, val2, val3 = struct.unpack("<HHHH", payload)
                    return [val0, val1, val2, val3][:count]

    # -----------------------------
    # 5. INTERNAL HELPERS (PRIVATE)
    # -----------------------------

    def _get_default_lowpass(self, rate):
        """
        Return the largest supported LPF cutoff frequency that satisfies Nyquist (≤ 0.4 * rate).
        If no such frequency exists, returns the minimum supported LPF and prints a warning.
        """

        # Find the largest supported LPF cutoff frequency that satisfies Nyquist (≤ 0.4 * rate)
        target = 0.4 * rate
        valid = [f for f in self._LPF_MAP if f <= target]
        if valid:
            return max(valid)

        # If no valid LPF cutoff exists, return the minimum supported and print a warning
        min_supported = min(self._LPF_MAP)
        print(
            f"\nWarning: Minimum supported lowpass filter cutoff selected ({min_supported} Hz). This cutoff violates\n"
            f"         Nyquist for the requested rate ({rate} Hz) and aliasing will pollute your data.\n"
        )
        return min_supported

    def _check_bandwidth(self, num_outputs, rate, baud):
        """
        Check if the requested configuration exceeds the physical serial line capacity.

        Formula:
          usage = 11bytes x num_outputs x output_rate (kB/s)
          capacity = baudrate (kB/s)  -- Note: normalized by 10 bits/byte and 1000 bytes/kB
        """
        usage_kb = (11.0 * num_outputs * rate) / 1000.0
        capacity_kb = (baud / 10.0) / 1000.0
        ratio = usage_kb / capacity_kb

        formula_str = (
            f"Formula:\n"
            f"  usage = 11bytes x {num_outputs} x {rate} = {usage_kb:.2f} (kB/s)\n"
            f"  capacity = {baud} / 10 = {capacity_kb:.2f} (kB/s)"
        )

        explanation = (
            "To resolve this bandwidth issue, please try the following options in order:\n"
            "  1) Increase the baudrate if possible (e.g. using set_baudrate)\n"
            "  2) Reduce the output rate (e.g. using set_streaming_rate)"
        )

        if ratio > 0.75:
            raise ValueError(
                f"Requested configuration exceeds serial line bandwidth capacity (ratio: {ratio:.1%}).\n"
                f"{formula_str}\n"
                f"{explanation}"
            )
        elif ratio > 0.50:
            print(f"Requested configuration consumes >50% of serial line capacity (ratio: {ratio:.1%}).\n{formula_str}\n")

    # ---------------------------
    # 6. PARSING THREAD (PRIVATE)
    # ---------------------------

    def _host_side_parsing_thread(self):
        """
        Background parsing thread execution loop.
        Drains the OS buffer and parses incoming bytes.
        """
        buffer = bytearray()
        received_this_cycle = set()

        try:
            while self._listening:
                if self._connection.in_waiting > 0:
                    buffer.extend(self._connection.read(self._connection.in_waiting))
                    while True:
                        result = self._next_packet(buffer)  # retrieve packet from buffer
                        if result is None:
                            break
                        ptype, payload = result
                        self._parse_packet_payload(ptype, payload)  # parse packet payload
                        received_this_cycle.add(ptype)
                        if ptype == self._trigger_packet_id:
                            if self._packet_ids.issubset(received_this_cycle):
                                self._commit_data()  # commits timestamped, parsed data to self._latest
                            received_this_cycle.clear()
                else:
                    time.sleep(0.5 / self.rate)
        except Exception as e:
            print(f"\n[Parsing thread crash] {type(e).__name__}: {e}")

    def _next_packet(self, buffer):
        """
        Pop the next valid 11-byte WitMotion packet from buffer.
        Returns (ptype, payload) or None if no complete valid packet is available.
        """
        while len(buffer) >= 11:
            if buffer[0] != self._FRAME_HEADER:
                buffer.pop(0)
                continue
            packet = buffer[:11]
            if sum(packet[:-1]) & 0xFF != packet[-1]:
                buffer.pop(0)
                continue
            del buffer[:11]
            return packet[1], bytes(packet[2:10])
        return None

    def _parse_packet_payload(self, ptype, payload):
        """
        Parses raw 8-byte data payloads depending on packet type.

        The hardware streams range-scaled 16-bit integers. This method converts them
        to SI units at the driver boundary: acceleration to m/s² (using _STANDARD_GRAVITY
        = 9.80665 m/s²), angular velocity to rad/s, magnetic field to µT, and Euler angles
        to rad. Callers receive SI values exclusively; no further conversion is expected.
        """
        if ptype == 0x50:  # Time Packet
            _, _, _, hh, mi, ss, ms = struct.unpack("<BBBBBBH", payload)
            self._sensor_time_ms = ms + ss * 1000 + mi * 60000 + hh * 3600000  # convert chip datetime to milliseconds

        elif ptype == 0x51:  # Acceleration & Temperature
            assert self.acc_range is not None, "acc_range not set — did connect() succeed?"
            ax, ay, az, temp = struct.unpack("<hhhh", payload)
            scale = self.acc_range / 32768.0 * self._STANDARD_GRAVITY
            lst = self._sensor_data["acc"]
            lst[0] = round(ax * scale, self._acc_ndigits)
            lst[1] = round(ay * scale, self._acc_ndigits)
            lst[2] = round(az * scale, self._acc_ndigits)
            self._sensor_data["temp"] = round(temp / 100.0, 2)

        elif ptype == 0x52:  # Angular Velocity
            assert self.gyro_range is not None, "gyro_range not set — did connect() succeed?"
            gx, gy, gz, _ = struct.unpack("<hhhh", payload)
            scale = self.gyro_range / 32768.0 * (math.pi / 180.0)
            lst = self._sensor_data["gyro"]
            lst[0] = round(gx * scale, self._gyro_ndigits)
            lst[1] = round(gy * scale, self._gyro_ndigits)
            lst[2] = round(gz * scale, self._gyro_ndigits)

        elif ptype == 0x53:  # Angle (Euler) — ±180° full scale → ±π rad
            roll, pitch, yaw, _ = struct.unpack("<hhhh", payload)
            lst = self._sensor_data["angle"]
            lst[0] = round(roll / 32768.0 * math.pi, 5)
            lst[1] = round(pitch / 32768.0 * math.pi, 5)
            lst[2] = round(yaw / 32768.0 * math.pi, 5)

        elif ptype == 0x54:  # Magnetic Field — 0.01 µT/LSB (fixed range)
            mx, my, mz, _ = struct.unpack("<hhhh", payload)
            lst = self._sensor_data["mag"]
            lst[0] = round(mx * 0.01, 2)
            lst[1] = round(my * 0.01, 2)
            lst[2] = round(mz * 0.01, 2)

        elif ptype == 0x59:  # Quaternions — scalar-first [w, x, y, z] matching hardware field order
            q0, q1, q2, q3 = struct.unpack("<hhhh", payload)
            lst = self._sensor_data["quat"]
            lst[0] = round(q0 / 32768.0, 5)  # w (scalar)
            lst[1] = round(q1 / 32768.0, 5)  # x
            lst[2] = round(q2 / 32768.0, 5)  # y
            lst[3] = round(q3 / 32768.0, 5)  # z

    def _commit_data(self):
        """
        Updates Software PLL timestamp, hardware time unwrapping, and dispatches data.
        """
        t_pc = time.perf_counter()
        if self._start_pc_time is None:
            self._start_pc_time = t_pc

        t_elapsed_pc = t_pc - self._start_pc_time

        # Unwrapped Monotonic Hardware Clock using day-based monotonic ms
        t_s = None
        if self._sensor_time_ms is not None:
            if self._start_sensor_ms is None:
                self._start_sensor_ms = self._sensor_time_ms
            t_s = (self._sensor_time_ms - self._start_sensor_ms) / 1000.0

        # Update Software PLL timeline
        if self._pll_t_est is None:
            self._pll_t_est = t_elapsed_pc
            pll_s = self._pll_t_est
        else:
            dt_raw = t_elapsed_pc - self._pll_t_est
            # Dynamic step sizing for cycle-slip robustness
            steps = round(dt_raw / self._pll_sample_period)
            if steps < 1:
                steps = 1
            t_pred = self._pll_t_est + steps * self._pll_sample_period
            self._pll_t_est = self._pll_alpha * t_elapsed_pc + (1.0 - self._pll_alpha) * t_pred
            pll_s = self._pll_t_est

        # Save timestamps in data model
        self._sensor_data["time_s"] = t_s
        self._sensor_data["pll_time_s"] = pll_s

        # Store a copy of the completed frame — only keys with data (excludes disabled outputs).
        # Lists are shallow-copied so the consumer's frame is not aliased to the live buffer.
        data_copy = {k: list(v) if isinstance(v, list) else v for k, v in self._sensor_data.items() if v is not None}
        with self._queue_lock:
            self._latest = data_copy
