# Error-State Kalman Filter for Kinematic State Estimation of CMU Prosthesis Leg

For general introduction of Error State Kalman Filter in IMU
navigation problems consult Joan Sola, "Quaternion kinematics for
the error-state Kalman filter" (2017, arXiv:1711.02508v1)

Follow's Haosen (Russell) Xing's implementation of ESKF.

## Overview

The filter fuses two IMUs (thigh and foot) with the joint encoders to estimate the pose of the leg. It tracks two rigid bodies, the thigh and the foot. Each body carries an attitude quaternion, a gyro bias, a position, a velocity, and an accelerometer bias, which is 15 error states per body and 30 in total.

`StateEstimator` owns the filter and is its only caller. It runs once per control tick.

## State and layout

The nominal state is a `NominalState` per body, propagated with the full nonlinear equations. The Kalman filter tracks the error between that nominal state and the truth, and after each correction the error is injected into the nominal state and reset to zero.

The error state is ordered `[dtheta, gyro bias, position, velocity, accel bias]` per body, thigh first then foot. `P`, `A`, `Q`, and every measurement Jacobian in `measurements.py` use that same order.

The attitude error is local, so `R_true = R_nom (I + hat(dtheta))` and injection right multiplies the quaternion. The bench Simulink model uses the global convention and its `A` is not interchangeable with this one.

## Each tick

1. Remap both IMUs from sensor axes into body axes.
2. Compute `dt` from the caller's clock.
3. For the first `cfg.init_duration` seconds, run the complementary filter on each IMU to align attitude with gravity. Estimates are not valid during this window.
4. After that, predict, then correct, then inject.

`healthy` goes false when an IMU stream has gone stale past `cfg.imu_stale_timeout`. The filter keeps running on the held sample.

## Measurements

Three corrections run in sequence each tick. Each one returns a residual, a Jacobian, and a measurement covariance, and is applied as its own update.

**Zero velocity update.** When the foot is detected to be standing still, its velocity is asserted to be zero. This is the only correction that anchors absolute drift. `ZUPT_AdaptiveGLRT` in `detectors.py` decides when the foot is stationary, and the detector only sees fresh foot samples.

**Position loop closure.** The encoders give the position of the thigh relative to the foot through forward kinematics. Adding that to the foot position predicts the thigh position, and the difference from the current thigh estimate is the residual. Only the thigh is corrected. The foot pose is treated as a pseudo measurement, so its uncertainty is pulled from `P` into the measurement covariance.

**Velocity loop closure.** Same idea for velocity. The foot body twist is composed with the leg body Jacobian and the joint rates to predict the thigh velocity. Only the thigh is corrected.

The forward kinematics and Jacobians come from `kinematics.py`, which is the symbolic derivation and the source of truth. `measurements.py` holds the numpy transcription of it.

## Configuration

All tuning lives in `ESKFConfig` in `config/params.py`: process noise sigmas, measurement noise sigmas, initial covariance, complementary filter settings, and the `dt` clamp. `Q` is rebuilt every tick because it scales with `dt`. Values are inherited from the bench model and are starting points.

## Not done yet

- Link lengths and the foot IMU to ankle offset in `measurements.py` are placeholders.
- The encoder to chain angle sign is confirmed for the knee, not for the ankle.
- The sensor to body mount rotations encode the bench mounting and need checking on hardware.
- The stationarity detector constants are hard coded rather than in config.
- Relative yaw between the two bodies is not aligned at handoff.
