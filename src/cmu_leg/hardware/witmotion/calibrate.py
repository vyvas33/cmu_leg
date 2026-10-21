#!/usr/bin/env python3
"""
Interactive command-line calibration for the WitMotion HWT906: accelerometer
or magnetometer, with step-by-step on-screen guidance and keyboard-driven
start/stop control.

Usage:
    python3 -m drivers.witmotion.calibrate
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from drivers.witmotion import WitMotionSensor, discover_port
from pdreckon.config import MAG_FIELD_PGH_JUL_2026

ACC_GUIDANCE = """
Accelerometer calibration determines the sensor's zero-offset so that it
reads exactly 1g straight down when at rest.

Procedure:
  1. Place the sensor on a flat, stable, motionless surface (e.g. a table or
     the floor) -- not held in your hand, since even a slight hand tremor
     will throw off the calibration.
  2. Make sure nothing is vibrating nearby (no running motors, no one
     bumping the table) for the duration of the calibration.
  3. Do not touch or move the sensor once calibration starts.

Calibration takes about 6 seconds and runs automatically once started --
there is nothing else to do but keep the sensor still.
"""

MAG_GUIDANCE = """
Magnetometer calibration determines and removes local magnetic distortions
(from nearby metal, electronics, or the sensor's own mounting hardware) so
it can accurately measure the direction of Earth's magnetic field.

Before starting:
  - Move at least 1-2 meters away from large metal objects, motors, speakers,
    and other electronics. Calibration "learns" whatever magnetic
    environment you calibrate in, so a distorted environment now means
    distorted headings later.

Procedure (rotate SLOWLY -- about one full 360 degree turn every ~5 seconds,
no fast snaps or jerky motion):
  1. Hold the sensor with one axis pointing straight up. Slowly rotate it a
     full 360 degrees about that (vertical) axis, taking about 5 seconds.
  2. Tip the sensor so a DIFFERENT axis now points up, and repeat: one slow
     360 degree rotation about that new vertical axis.
  3. Tip again so the THIRD axis points up, and repeat once more.
  4. Finally, spend another 15-20 seconds slowly tumbling the sensor through
     random, varied orientations -- not just clean axis spins. This fills in
     the gaps between the three rotations above, since the fitting algorithm
     needs coverage across the whole sphere of possible orientations, not
     just three circles on it.

Total calibration time: roughly 30-45 seconds of continuous slow motion.
Avoid holding the sensor still at any single orientation for more than a
second or two.
"""

MAG_SCALE_DURATION_S = 40.0

MAG_SCALE_GUIDANCE = f"""
Host-side magnetometer scale calibration. The sensor's on-board magnetic
calibration (CALSW spherical fit) only removes hard-iron offset -- it does
not correct the overall magnitude/gain of the readings. This instead
records raw data and fits both an offset and a scale factor against Earth's
known local field strength, correcting magnitude as well as direction.

Before starting:
  - Move at least 1-2 meters away from large metal objects, motors, speakers,
    and other electronics.

Procedure (rotate SLOWLY -- about one full 360 degree turn every ~5 seconds,
no fast snaps or jerky motion):
  1. Hold the sensor with one axis pointing straight up. Slowly rotate it a
     full 360 degrees about that (vertical) axis, taking about 5 seconds.
  2. Tip the sensor so a DIFFERENT axis now points up, and repeat.
  3. Tip again so the THIRD axis points up, and repeat once more.
  4. Finally, spend the remaining time slowly tumbling the sensor through
     random, varied orientations -- not just clean axis spins.

Recording starts immediately and runs automatically for {MAG_SCALE_DURATION_S:.0f} seconds.
"""


def _wait_for_enter(prompt):
    input(f"{prompt} (press Enter) ")


def _fit_sphere(points):
    """
    Least-squares fit of a sphere to 3D points, linearizing
    |p - center|^2 = radius^2 into 2*p.center + (radius^2 - |center|^2) = |p|^2.

    Returns (center, radius).
    """
    A = np.hstack([2 * points, np.ones((points.shape[0], 1))])
    b = np.sum(points**2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = np.sqrt(sol[3] + np.sum(center**2))
    return center, radius


def calibrate_accelerometer(sensor):
    print(ACC_GUIDANCE)
    _wait_for_enter("When the sensor is flat and completely still, start calibration")
    sensor.calibrate_accelerometer()
    sensor.save_config()


def calibrate_magnetometer(sensor):
    print(MAG_GUIDANCE)
    _wait_for_enter("When ready to begin rotating, start calibration")
    sensor.start_magnetic_calibration()
    _wait_for_enter("Rotate the sensor now. When finished, stop calibration")
    sensor.stop_magnetic_calibration()
    sensor.save_config()


def calibrate_magnetometer_scale(sensor):
    """
    Record a raw magnetometer rotation sweep and fit a host-side offset +
    scale correction against Earth's known local field magnitude. Saves the
    result to output/calib_result_<timestamp>.json.
    """
    print(MAG_SCALE_GUIDANCE)
    _wait_for_enter("When ready to begin rotating, start recording")

    sensor.set_streaming_rate(100)
    sensor.set_outputs(ts=False, acc=True, gyro=False, mag=True)
    sensor.stream()
    sensor.drain_data_buffer()  # discard anything buffered before recording starts

    frames = []
    start_time = time.perf_counter()
    update_time = 0.0
    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed > MAG_SCALE_DURATION_S:
            break
        if elapsed > update_time:
            print(f"Recording: {elapsed:.0f}s / {MAG_SCALE_DURATION_S:.0f}s")
            update_time += 1.0
        frames.extend(sensor.drain_data_buffer())
        time.sleep(0.5 / sensor.rate)
    frames.extend(sensor.drain_data_buffer())
    sensor.stop_streaming()

    mag = np.array([f["mag"] for f in frames])
    acc = np.array([f["acc"] for f in frames])

    offset, raw_radius = _fit_sphere(mag)
    expected_uT = float(np.linalg.norm(MAG_FIELD_PGH_JUL_2026) * 100)  # G -> uT
    scale = expected_uT / raw_radius

    corrected = (mag - offset) * scale
    corrected_norm = np.linalg.norm(corrected, axis=1)

    # rotation-invariant acc/mag angle check: should hold regardless of orientation
    a_n = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    m_n = corrected / np.linalg.norm(corrected, axis=1, keepdims=True)
    angle_deg = np.degrees(np.arccos(np.clip(np.sum(a_n * m_n, axis=1), -1, 1)))

    print(f"\n{len(frames)} samples recorded.")
    print(f"Fitted offset:  [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}] uT (raw driver units)")
    print(f"Fitted radius:  {raw_radius:.2f} uT (raw driver units) -> scale factor: {scale:.4f}")
    print(
        f"Corrected magnitude: median={np.median(corrected_norm):.2f} uT, "
        f"std={np.std(corrected_norm):.2f} uT (target: {expected_uT:.2f} uT)"
    )
    print(f"acc/mag angle spread (rotation-invariant check): std={np.std(angle_deg):.2f} deg")

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sensor_port": sensor.port,
        "n_samples": len(frames),
        "duration_s": MAG_SCALE_DURATION_S,
        "expected_field_uT": expected_uT,
        "offset_uT": offset.tolist(),
        "scale": float(scale),
        "fitted_radius_uT": float(raw_radius),
        "corrected_median_uT": float(np.median(corrected_norm)),
        "corrected_std_uT": float(np.std(corrected_norm)),
    }

    os.makedirs("output", exist_ok=True)
    fname = f"output/calib_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCalibration result saved to '{fname}'")


def main():
    choice = (
        input(
            "Calibrate (a)ccelerometer, (m)agnetometer offset [on-device], "
            "or (s)cale-correct magnetometer [host-side]? [a/m/s]: "
        )
        .strip()
        .lower()
    )
    if choice not in ("a", "m", "s"):
        print(f"Unrecognized choice: {choice!r}. Expected 'a', 'm', or 's'.")
        sys.exit(1)

    sensor = None
    try:
        sensor = WitMotionSensor(port=discover_port(), baud_bps=460800)
        sensor.connect()

        if choice == "a":
            calibrate_accelerometer(sensor)
        elif choice == "m":
            calibrate_magnetometer(sensor)
        else:
            calibrate_magnetometer_scale(sensor)

    except (ConnectionError, RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    finally:
        if sensor is not None:
            sensor.disconnect()


if __name__ == "__main__":
    main()
