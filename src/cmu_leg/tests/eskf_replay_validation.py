"""Validate our ESKF against the baseline Simulink ESKF on real hardware data.

Drives `estimation.eskf.ESKF` with the IMU + encoder signals logged in a baseline
level-walking trial and overlays our `hip_angle` against the baseline ESKF's own
hip_angle from the SAME file (identical inputs => any difference is our filter,
not the data). There is no Vicon in this log (only a sync channel), so the
baseline ESKF output is the reference here; swap in mocap ground truth later.

Data: data/gaitData20251015_test7_NN_vicon.mat (Oct 2025, CMU leg, 1 kHz, ~31 s).
Signal columns (0-based into scope_data.base.data) were triangulated from the
embedded signalNames, plot_gait_data.m, and a gravity-magnitude check:

    hip_angle (baseline ESKF, rad) .. 0
    thigh gyro [rad/s] ............. 42,43,44      thigh accel [m/s^2] .. 45,46,47
    foot  accel [m/s^2] ........... 48,49,50       foot  gyro  [rad/s] .. 51,52,53
    knee angle [rad] .............. 7               ankle angle [rad] .... 15
    time [s] ...................... 107

Run:  uv run python -m cmu_leg.tests.eskf_replay_validation
Writes eskf_replay_validation.png next to this file.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmu_leg.types.sensors import IMUFrame
from cmu_leg.types.state import ProsthesisState
from cmu_leg.estimation.eskf.eskf import ESKF, R_BODY_SENSOR_FO, _PERM_TH

HERE = Path(__file__).parent
MAT = HERE / "data" / "gaitData20251015_test7_NN_vicon.mat"
PNG = HERE / "eskf_replay_validation.png"

COL = dict(hip=0, thigh_gyro=[42, 43, 44], thigh_acc=[45, 46, 47],
           foot_acc=[48, 49, 50], foot_gyro=[51, 52, 53],
           knee_ang=7, ankle_ang=15, time=107)

INIT_SKIP_S = 6.0   # baseline ESKF comp-filter init; exclude from metrics


def load_base(mat_path=MAT):
    """Return the (N, 108) base signal matrix from the trial .mat."""
    return sio.loadmat(str(mat_path), squeeze_me=True,
                       struct_as_record=False)["scope_data"].base.data


def replay(D, target_hz=500.0):
    """Drive our ESKF with the logged IMU + encoders. Returns dict of traces.

    The log is 1 kHz, but the ESKF runs at the control rate on hardware
    (~500 Hz, runtime/control.py). Since the process noise is (sigma*dt)^2 --
    rate-dependent -- the filter must be validated at the SAME dt it will
    integrate with, or the sigmas tuned for 500 Hz inject the wrong Q here.
    So decimate the log to target_hz. Walking is <~10 Hz and the baseline data
    is already LPF'd (Nyquist 250 Hz after /2), so plain decimation won't alias.
    """
    t_full = D[:, COL["time"]]
    native_hz = 1.0 / np.median(np.diff(t_full))
    decim = max(1, round(native_hz / target_hz))
    D = D[::decim]

    t = D[:, COL["time"]]
    N = len(t)
    # The baseline Read IMU block outputs are ALREADY in EKF/body frame (confirmed
    # in nm_nn_swing: the Mux+Selector axis reorder happens inside the block, and
    # the logged Gain* taps are its output ports -- annotated "For EKF x-fwd,
    # y-left, z-up"). Our ESKF expects RAW SENSOR and applies the permutation
    # itself (_PERM_TH / R_BODY_SENSOR_FO), so we undo the baseline's permutation
    # first (raw = P^T @ logged) to avoid a double reorder. (The double reorder
    # only flips x/y, so it is invisible at standing but inverts the swing.)
    ta = (_PERM_TH.T @ D[:, COL["thigh_acc"]].T).T
    tg = (_PERM_TH.T @ D[:, COL["thigh_gyro"]].T).T
    fa = (R_BODY_SENSOR_FO.T @ D[:, COL["foot_acc"]].T).T
    fg = (R_BODY_SENSOR_FO.T @ D[:, COL["foot_gyro"]].T).T
    ka, aa = D[:, COL["knee_ang"]], D[:, COL["ankle_ang"]]
    kv, av = np.gradient(ka, t), np.gradient(aa, t)   # joint vels (logged deriv is spiky)
    dt_nom = float(np.median(np.diff(t)))
    dt_arr = np.diff(t, prepend=t[0] - dt_nom)

    eskf = ESKF()
    state = ProsthesisState()
    hip = np.full(N, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):  # ZUPT detector norm==0 at rest
        for k in range(N):
            dt = dt_arr[k] if dt_arr[k] > 1e-5 else dt_nom
            imu = IMUFrame(
                thigh_acc=tuple(ta[k]), thigh_gyro=tuple(tg[k]),
                foot_acc=tuple(fa[k]), foot_gyro=tuple(fg[k]),
                thigh_time=t[k], foot_time=t[k], thigh_fresh=True, foot_fresh=True)
            state.knee_angle, state.ankle_angle = ka[k], aa[k]
            state.knee_velocity, state.ankle_velocity = kv[k], av[k]
            eskf.update(imu, state, t[k], dt)
            hip[k] = state.hip_angle * 180 / np.pi

    return dict(t=t, hip_ours=hip, hip_base=D[:, COL["hip"]] * 180 / np.pi,
                thigh_gyro=tg, thigh_acc=ta)


def _rms(a, b, m):
    return float(np.sqrt(np.nanmean((a[m] - b[m]) ** 2)))


def report(r):
    t, ho, hb = r["t"], r["hip_ours"], r["hip_base"]
    static = (t > INIT_SKIP_S) & (t < 15.0)
    walk = t > 16.0
    print(f"samples={len(t)}  duration={t[-1]:.1f}s")
    print(f"  static (6-15s)  RMS = {_rms(ho, hb, static):6.2f} deg   (residual offset -- thigh mount-tilt stage TBD)")
    print(f"  walking (>16s)  RMS = {_rms(ho, hb, walk):6.2f} deg   (phase OK after frame fix; residual = real ESKF diff)")
    print(f"  hip range  ours {np.nanmin(ho):.1f}..{np.nanmax(ho):.1f}   base {np.nanmin(hb):.1f}..{np.nanmax(hb):.1f} deg")


def plot(r, png=PNG):
    t, ho, hb = r["t"], r["hip_ours"], r["hip_base"]
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))

    # (1) our vs baseline hip
    ax[0].plot(t, hb, color="tab:red", lw=1.8, label="baseline Simulink ESKF")
    ax[0].plot(t, ho, color="black", lw=1.1, label="our Python ESKF")
    ax[0].axvspan(0, INIT_SKIP_S, color="0.9", label="init (excluded)")
    ax[0].set_ylabel("hip angle [deg]")
    ax[0].set_title("Our ESKF vs baseline ESKF — identical IMU (Oct-2025 CMU-leg level walking)")
    ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)

    # (2) error -- init region masked out so the (converging) transient doesn't
    # dominate the y-range; scaled to the post-init error only.
    err = ho - hb
    err_post = np.where(t >= INIT_SKIP_S, err, np.nan)
    ax[1].plot(t, err_post, color="tab:blue", lw=0.8)
    ax[1].axhline(0, color="0.5", lw=0.7); ax[1].axvspan(0, INIT_SKIP_S, color="0.9")
    ax[1].set_ylabel("difference (ours - base) [deg]"); ax[1].set_xlabel("time [s]")
    ax[1].set_title("Difference")
    lim = 1.15 * np.nanmax(np.abs(err_post))
    ax[1].set_ylim(-lim, lim); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print(f"wrote {png}")


def main():
    r = replay(load_base())
    report(r)
    plot(r)


if __name__ == "__main__":
    main()
