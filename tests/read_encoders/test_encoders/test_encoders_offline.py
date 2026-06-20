"""Offline verification of read_encoders against a real SLRT recording.

Feeds the raw count columns from ``knee.mat`` / ``ankle.mat`` (captured from the
Simulink ``read_encoders`` File scopes) through the Python ``read_encoders()``
and checks every engineering output matches the recorded engineering column
bit-exactly.

The conversion is only as good as the calibration it's given, so the session's
own calibration (the encoder zeros + counts->rad polynomials that were live on
the prosthesis machine when the recording was captured) is loaded from
``tests/calibration/`` and injected. These came from prosthesis_control_NPC
(spring_zero_positions.mat dated Sep-2025) and reproduce the recording to
machine precision. See memory: encoder-zero-provenance.

Recording column order (confirmed from the .mat signalNames):
  knee.mat : [knee_motor_angle, knee_angle, knee_motor_counts(raw),
              knee_joint_counts(raw), knee_torque, time]
  ankle.mat: [ankle_motor_angle, ankle_spring_angle, ankle_angle,
              ankle_motor_counts(raw), ankle_joint_counts(wrapped,raw),
              ankle_torque, time]
"""
from __future__ import annotations

import sys
from math import pi
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

# This test lives inside the read_encoders package dir; put tests/ on the path
# so `import read_encoders` resolves to the module package one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from read_encoders import EncoderCalibration, RawSensorFrame, read, read_encoders

DATA_DIR = Path(__file__).parent                        # knee.mat / ankle.mat sit beside this test
CAL_DIR = Path(__file__).resolve().parents[2] / "calibration"

# --- recording column indices ------------------------------------------------
K_MOTOR_ANGLE, K_ANGLE, K_MOTOR_CNT, K_JOINT_CNT, K_TORQUE = 0, 1, 2, 3, 4
A_MOTOR_ANGLE, A_SPRING_ANGLE, A_ANGLE, A_MOTOR_CNT, A_JOINT_CNT, A_TORQUE = 0, 1, 2, 3, 4, 5

# Bit-exact tolerance (float64 round-trip through the conversion chain).
TOL = 1e-9


def _load_session_calibration() -> EncoderCalibration:
    """Build EncoderCalibration from the recording session's .mat files,
    mapping variables exactly as baseline_controller/params.m does."""
    z = sio.loadmat(CAL_DIR / "spring_zero_positions.mat")
    p = sio.loadmat(CAL_DIR / "encoder_counts_to_rad_polys.mat")
    a = sio.loadmat(CAL_DIR / "ankle_angle_offset.mat")
    f = lambda m, k: float(np.squeeze(m[k]))
    i = lambda m, k: int(np.squeeze(m[k]))
    t = lambda m, k: tuple(np.squeeze(m[k]).tolist())
    return EncoderCalibration(
        motor_counts_to_rad=2 * pi / 2 ** 30,
        knee_motor_offset=i(z, "kneemotor0pos"),
        ankle_motor_offset=i(z, "anklemotor0pos"),
        knee_counts_to_rad=f(p, "knee_counts_to_rad"),
        knee_joint_offset=i(z, "knee0pos"),
        knee_counts_mean=f(p, "knee_encoder_counts_mean"),
        knee_counts_std=f(p, "knee_encoder_counts_std"),
        knee_err_poly=t(p, "knee_angle_err_poly"),
        knee_err_mean=f(p, "knee_angle_err_mean"),
        knee_err_std=f(p, "knee_angle_err_std"),
        ankle_counts_to_rad=2 * pi / 2 ** 17,
        ankle_joint_offset=i(z, "ankle0pos"),
        ankle_angle_offset=f(a, "ankle_angle_offset"),  # carried, not applied here
        ankle_spring_counts_to_rad_poly=t(p, "ankle_spring_counts_to_rad_poly"),
        ankle_spring_counts_mean=f(p, "ankle_spring_encoder_counts_mean"),
        ankle_spring_counts_std=f(p, "ankle_spring_encoder_counts_std"),
        ankle_spring_err_poly=t(p, "ankle_spring_angle_err_poly"),
        ankle_spring_err_mean=f(p, "ankle_spring_angle_err_mean"),
        ankle_spring_err_std=f(p, "ankle_spring_angle_err_std"),
    )


# --- fixtures -----------------------------------------------------------------
@pytest.fixture(scope="module")
def knee():
    return sio.loadmat(DATA_DIR / "knee.mat")["data"]


@pytest.fixture(scope="module")
def ankle():
    return sio.loadmat(DATA_DIR / "ankle.mat")["data"]


@pytest.fixture(scope="module")
def readings(knee, ankle):
    """Run read_encoders over every recorded sample with the session calibration.
    Knee and ankle came from separate scopes on the same run; pair them row-by-row
    into one RawSensorFrame per tick.
    """
    cal = _load_session_calibration()
    n = min(knee.shape[0], ankle.shape[0])
    out = {k: np.empty(n) for k in (
        "knee_motor_angle", "knee_angle", "knee_torque",
        "ankle_motor_angle", "ankle_spring_angle", "ankle_angle", "ankle_torque",
    )}
    for idx in range(n):
        frame = RawSensorFrame(
            knee_motor_counts=int(knee[idx, K_MOTOR_CNT]),
            knee_joint_counts=int(knee[idx, K_JOINT_CNT]),
            ankle_motor_counts=int(ankle[idx, A_MOTOR_CNT]),
            ankle_joint_counts=int(ankle[idx, A_JOINT_CNT]),
        )
        s = read_encoders(frame, cal)
        for k in out:
            out[k][idx] = getattr(s, k)
    return out


def _max_err(pred, rec):
    return float(np.max(np.abs(pred - rec)))


# --- every read_encoders output, bit-exact vs the recording -------------------
def test_knee_motor_angle(readings, knee):
    assert _max_err(readings["knee_motor_angle"], knee[:, K_MOTOR_ANGLE]) < TOL


def test_knee_angle(readings, knee):
    assert _max_err(readings["knee_angle"], knee[:, K_ANGLE]) < TOL


def test_knee_torque(readings, knee):
    assert _max_err(readings["knee_torque"], knee[:, K_TORQUE]) < TOL


def test_ankle_motor_angle(readings, ankle):
    assert _max_err(readings["ankle_motor_angle"], ankle[:, A_MOTOR_ANGLE]) < TOL


def test_ankle_spring_angle(readings, ankle):
    assert _max_err(readings["ankle_spring_angle"], ankle[:, A_SPRING_ANGLE]) < TOL


def test_ankle_angle(readings, ankle):
    assert _max_err(readings["ankle_angle"], ankle[:, A_ANGLE]) < TOL


def test_ankle_torque(readings, ankle):
    assert _max_err(readings["ankle_torque"], ankle[:, A_TORQUE]) < TOL


# --- I/O boundary -------------------------------------------------------------
def test_read_is_stubbed():
    with pytest.raises(NotImplementedError):
        read()
