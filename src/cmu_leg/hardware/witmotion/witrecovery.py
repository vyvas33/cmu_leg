"""
WitMotion HWT906 Communication Recovery Utility.

Run this script when the sensor's baud rate is unknown or communication is lost.

Step 1 — Discovery: probe each known baud rate via discover_baud().
Step 2 — Recovery: if step 1 fails, blindly send 'set baud to 9600' at each candidate
         starting from the highest, then save to flash at 9600 and verify.

Usage:
    python witrecovery.py

hgeyer, June 2026
"""

import sys
import time

import serial
from witapi import WitMotionSensor, discover_baud, discover_port


def discover_comm(port=None):
    """
    Discover or recover communication with a WitMotion HWT906 sensor.
    Returns (port, baud) at which communication was established.

    Raises:
        FileNotFoundError: No CH340 device found.
        ValueError: Multiple CH340 devices found.
        RuntimeError: Recovery failed at all baud rates.
    """
    if port is None:
        port = discover_port()
        print(f"Found CH340 on port {port}.\n")

    # Step 1: discovery
    print("Probing baud rates ...")
    detected = None
    try:
        detected = discover_baud(port)
        print(f"  Detected at {detected} bps.")
    except ConnectionError:
        print("  No response at any baud rate.")

    # Step 2: blind recovery to 9600
    if detected is None:
        print("\nAttempting blind recovery to 9600 bps ...")
        for send_baud in sorted((b for b in WitMotionSensor._BAUD_MAP if b != 9600), reverse=True):
            if _blind_set_to_9600(port, send_baud):
                detected = 9600
                break
        if detected is None:
            raise RuntimeError(
                "Recovery failed. Sensor unresponsive at all baud rates. Check USB connection and sensor power."
            )

    print(" ")
    sensor = WitMotionSensor(port=port, baud_bps=detected)
    sensor.connect()
    sensor.show_config()
    sensor.disconnect()
    return port, detected


def _write_to_sensor(ser, byte_array):
    ser.write(WitMotionSensor._UNLOCK_CMD)
    time.sleep(0.05)
    ser.write(byte_array)
    time.sleep(0.05)


def _blind_set_to_9600(port, send_baud):
    """
    Send 'set baud to 9600' at send_baud, then reconnect at 9600 to save to flash.
    Baud change takes effect immediately, so flash save must go out at the new baud.
    """
    print(f"  Trying {send_baud} bps ...", end=" ", flush=True)
    try:
        with serial.Serial(port, send_baud, timeout=0.1) as ser:
            _write_to_sensor(ser, bytes([0xFF, 0xAA, 0x04, WitMotionSensor._BAUD_MAP[9600], 0x00]))
        # Sensor has switched to 9600 — reopen at 9600 to save to flash
        time.sleep(0.1)
        with serial.Serial(port, 9600, timeout=0.1) as ser:
            _write_to_sensor(ser, bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]))
        time.sleep(0.5)  # wait for flash write
    except serial.SerialException:
        print("port error.")
        return False

    try:
        discover_baud(port)
        print("success.")
        print("Recovery baudrate 9600 saved to sensor flash memory.")
        return True
    except ConnectionError:
        print("no response at 9600.")
        return False


if __name__ == "__main__":
    try:
        discover_comm()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)
