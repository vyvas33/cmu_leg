#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Characterization and Raw Performance Analysis Script

This utility module characterizes:
1. Yost Labs 3-Space IMU v2.1 (TSS)
2. WitMotion HWT906 IMU

It supports:
- Recording live IMU data to disk (.npz format)
- Loading existing recordings from disk
- Computing statistical metrics (mean, std, RMS, skewness, kurtosis, and outlier count)
- Generating and saving performance figures to the output directory:
  1. Time series plots
  2. Noise histograms with Gaussian fits
  3. Power Spectral Density (PSD) with sub-bin -3dB cutoff detection
  4. Allan Deviation (ADEV) with noise guide and key annotations
"""

import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
from cmu_leg.utility.utils import make_run_dir

# Setup search path so drivers can be resolved when executing directly
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# --- INIT ---

IMU_TYPE = "wit"  # "tss" (Yost Labs 3-Space) or "wit" (WitMotion) # Note that this script does not support the Yost IMU

if IMU_TYPE == "wit":
    from cmu_leg.hardware.witmotion.hwt906 import WitMotionSensor as ImuSensor
    from cmu_leg.hardware.witmotion.hwt906 import discover_port, discover_baud
else:
    print("IMU type unknown. Use either 'wit' or 'tss'")
    exit(1)


RECORD_DATA = False  # True to record live data; False to analyze existing data file
DURATION = 10.0  # [s]
DATA_FILE = "output/2026-08-05_155156_imu_char_fo/wit_data.npz"  # Path to an existing .npz recording; only used when RECORD_DATA is False


# Color system for plots (Tableau 10 soft colors)
COLOR_X = "#E15759"  # Soft Red
COLOR_Y = "#76B7B2"  # Soft Teal/Green
COLOR_Z = "#4E79A7"  # Soft Blue
AXIS_COLORS = [COLOR_X, COLOR_Y, COLOR_Z]
AXIS_LABELS = ["X-axis", "Y-axis", "Z-axis"]


# --- DATA RECORDING AND LOADING ---


def record_data(imu_type, duration, save_dir):
    """
    Connects to the specified IMU, configures streaming properties,
    collects data for a set duration, and saves it to a .npz file.

    Returns:
        Path to the saved .npz recording.
    """

    port = discover_port()
    baud_bps = discover_baud(port)
    sensor = ImuSensor(port=port, baud_bps=460800)

    try:
        sensor.connect()
        sensor.set_streaming_rate(500)
        sensor.set_lpf(188)
        sensor.set_outputs(ts=False, acc=True, gyro=True, ang=False, mag=True, quat=False)

        sensor.show_config()

        print(f"=== Recording IMU raw data for {IMU_TYPE.upper()} ===")

        sensor.stream()

        data_log = []
        start_time = None
        update_time = 0.0
        while True:
            batch = sensor.drain_data_buffer()
            if batch and start_time is None:
                start_time = time.perf_counter()

            if start_time is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed > duration:
                    break
                if elapsed > update_time:
                    print(f"Elapsed: {elapsed:.1f}s / {duration:.1f}s")
                    update_time += 1.0

            if batch:
                data_log.extend(batch)

            time.sleep(0.5 / sensor.rate)

        # Convert dictionary frames to 2D numpy array rows
        rows = []
        for frame in data_log:
            row = []
            for v in frame.values():
                row.extend(v) if isinstance(v, list) else row.append(v)
            rows.append(row)
        np_data = np.array(rows, dtype=float)

        data_path = save_dir / f"{imu_type}_data.npz"
        np.savez(data_path, data=np_data, header=np.array([sensor.get_data_header()]))
        print(f"Data saved to '{data_path}' [{np_data.shape[0]} frames, columns: {sensor.get_data_header()}]\n")
        return data_path

    except (ConnectionError, RuntimeError) as e:
        print(f"\nRecording Error: {e}")
        sys.exit(1)

    finally:
        if sensor is not None:
            sensor.stop_streaming()
            sensor.set_streaming_rate(0.5)
            sensor.disconnect()


def load_data(file_path):
    """
    Loads IMU data from an NPZ file containing 'data' and 'header' arrays.

    Returns:
        t: zeroed time array [s]
        acc: Nx3 array [m/s²]
        gyr: Nx3 array [rad/s]
        mag: Nx3 array [µT]
        columns: list of column headers
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Recording file not found: {file_path}")

    print(f"Loading data from {file_path}...")
    loaded = np.load(file_path, allow_pickle=True)
    np_data = loaded["data"]
    columns = [c.strip() for c in str(loaded["header"][0]).split(",")]

    def _ci(name):
        return columns.index(name)

    t = np_data[:, _ci("host_time_s [s]")]
    t -= t[0]
    acc = np_data[:, [_ci("acc_x [m/s²]"), _ci("acc_y [m/s²]"), _ci("acc_z [m/s²]")]]
    gyr = np_data[:, [_ci("gyro_x [rad/s]"), _ci("gyro_y [rad/s]"), _ci("gyro_z [rad/s]")]]
    mag = np_data[:, [_ci("mag_x [µT]"), _ci("mag_y [µT]"), _ci("mag_z [µT]")]]

    print(f"Loaded {len(t)} frames over {t[-1]:.2f} s")
    return t, acc, gyr, mag, columns


# --- NUMERICAL NOISE ANALYSIS HELPERS ---


def compute_robust_noise_metrics(sig):
    """
    Computes standard and robust statistics using Median Absolute Deviation (MAD)
    for outlier detection.
    """
    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    # Standard deviation estimate from MAD (for normal distribution)
    std_mad = 1.4826 * mad if mad > 0 else np.std(sig)

    # Filter outliers (points further than 5 * std_mad from the median)
    threshold = 5.0 * std_mad
    non_outliers = np.abs(sig - med) <= threshold
    trimmed_sig = sig[non_outliers]
    num_outliers = len(sig) - len(trimmed_sig)

    mean_raw = np.mean(sig)
    std_raw = np.std(sig)
    skew_raw = scipy.stats.skew(sig)
    kurt_raw = scipy.stats.kurtosis(sig)

    mean_trim = np.mean(trimmed_sig)
    std_trim = np.std(trimmed_sig)
    skew_trim = scipy.stats.skew(trimmed_sig)
    kurt_trim = scipy.stats.kurtosis(trimmed_sig)

    return {
        "raw": (mean_raw, std_raw, skew_raw, kurt_raw),
        "trimmed": (mean_trim, std_trim, skew_trim, kurt_trim),
        "num_outliers": num_outliers,
        "std_mad": std_mad,
        "trimmed_sig": trimmed_sig,
    }


def print_sensor_stats(sensor_name, data, units):
    """
    Prints standard and clean (outlier-filtered) statistics for X, Y, Z axes.
    """
    print(f"\n--- {sensor_name} Statistics ({units}) ---")
    axes = ["X", "Y", "Z"]
    print(
        f"{'Axis':<8} | {'Mean':<10} | {'Std Dev':<10} | {'RMS':<10} | {'Peak-to-Peak':<12} | {'Skewness':<10} | {'Kurtosis':<10} | {'Outliers (>5 MAD)'}"
    )
    print("-" * 105)
    for i, axis in enumerate(axes):
        sig = data[:, i]
        stats = compute_robust_noise_metrics(sig)

        mean_raw, std_raw, skew_raw, kurt_raw = stats["raw"]
        rms_raw = np.sqrt(np.mean(sig**2))
        p2p_raw = np.max(sig) - np.min(sig)
        num_outliers = stats["num_outliers"]

        if num_outliers > 0:
            print(
                f"{axis + ' (raw)':<8} | {mean_raw:>10.5f} | {std_raw:>10.5f} | {rms_raw:>10.5f} | {p2p_raw:>12.5f} | {skew_raw:>10.3f} | {kurt_raw:>10.3f} | {num_outliers}"
            )
            mean_trim, std_trim, skew_trim, kurt_trim = stats["trimmed"]
            trimmed_sig = stats["trimmed_sig"]
            rms_trim = np.sqrt(np.mean(trimmed_sig**2))
            p2p_trim = np.max(trimmed_sig) - np.min(trimmed_sig)
            print(
                f"{axis + ' (clean)':<8} | {mean_trim:>10.5f} | {std_trim:>10.5f} | {rms_trim:>10.5f} | {p2p_trim:>12.5f} | {skew_trim:>10.3f} | {kurt_trim:>10.3f} |"
            )
        else:
            print(
                f"{axis:<8} | {mean_raw:>10.5f} | {std_raw:>10.5f} | {rms_raw:>10.5f} | {p2p_raw:>12.5f} | {skew_raw:>10.3f} | {kurt_raw:>10.3f} | 0"
            )


# --- PLOTTING AND ANALYSIS VISUALIZATIONS ---


def apply_plot_style():
    """
    Sets global matplotlib parameters for publication-quality plots.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.alpha": 0.3,
            "figure.titlesize": 14,
        }
    )


def plot_time_series(t, acc, gyr, mag, rate_tag, output_dir, imu_name, prefix):
    """
    Plots the three axes for accelerometer, gyroscope, and magnetometer over time.
    """
    apply_plot_style()
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{imu_name} IMU Raw Sensor Readings", fontweight="bold")

    # Accelerometer
    for i, col in enumerate(AXIS_COLORS):
        axs[0].plot(t, acc[:, i], color=col, label=AXIS_LABELS[i], lw=1.2)
    axs[0].set_ylabel("Acceleration (m/s²)")
    axs[0].set_title("Accelerometer Raw Readings")
    axs[0].grid(True)
    axs[0].legend(loc="upper right", ncol=3)

    # Gyroscope
    for i, col in enumerate(AXIS_COLORS):
        axs[1].plot(t, gyr[:, i], color=col, label=AXIS_LABELS[i], lw=1.2)
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].set_title("Gyroscope Raw Readings")
    axs[1].grid(True)

    # Magnetometer
    for i, col in enumerate(AXIS_COLORS):
        axs[2].plot(t, mag[:, i], color=col, label=AXIS_LABELS[i], lw=1.2)
    axs[2].set_ylabel("Magnetic Field (µT)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Magnetometer Raw Readings")
    axs[2].grid(True)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_imu_time_series_{rate_tag}.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Saved: {fig_path}")
    plt.close(fig)


def plot_noise_histograms(acc, gyr, mag, rate_tag, output_dir, imu_name, prefix):
    """
    Plots noise distribution histograms and fits normal Gaussian curves.
    """
    apply_plot_style()
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f"{imu_name} Sensor Noise Distribution & Gaussian Fits", fontweight="bold")

    sensor_names = ["Accelerometer", "Gyroscope", "Magnetometer"]
    sensor_data = [acc, gyr, mag]
    sensor_units = ["m/s²", "rad/s", "µT"]
    axes_names = ["X", "Y", "Z"]

    for s_idx, (name, data, unit) in enumerate(zip(sensor_names, sensor_data, sensor_units)):
        for a_idx, ax_name in enumerate(axes_names):
            ax = axs[a_idx, s_idx]
            sig = data[:, a_idx]

            stats = compute_robust_noise_metrics(sig)
            mean_trim, std_trim, skew_trim, kurt_trim = stats["trimmed"]
            num_outliers = stats["num_outliers"]

            # Center noise about robust median
            noise = sig - np.median(sig)
            plot_range = (-5 * std_trim, 5 * std_trim)

            n, bins, patches = ax.hist(
                noise, bins=40, range=plot_range, density=True, alpha=0.6, color=AXIS_COLORS[s_idx], edgecolor="none"
            )

            # Normal Gaussian fit using trimmed std
            x_pdf = np.linspace(plot_range[0], plot_range[1], 200)
            pdf = scipy.stats.norm.pdf(x_pdf, 0.0, std_trim)
            ax.plot(x_pdf, pdf, "k--", lw=1.5, label="Normal Fit")

            ax.set_title(f"{name} {ax_name}-axis Noise")
            ax.set_xlabel(f"Noise Value ({unit})")
            ax.set_ylabel("Probability Density")
            ax.set_xlim(plot_range)
            ax.grid(True)

            skew_raw = scipy.stats.skew(sig)
            kurt_raw = scipy.stats.kurtosis(sig)

            textstr = "\n".join(
                (
                    r"$\sigma_{\mathrm{raw}}=%.4f$" % (np.std(sig),),
                    r"$\sigma_{\mathrm{clean}}=%.4f$" % (std_trim,),
                    r"$Skew_{\mathrm{raw}}=%.2f$" % (skew_raw,),
                    r"$Skew_{\mathrm{clean}}=%.2f$" % (skew_trim,),
                    r"$Kurt_{\mathrm{raw}}=%.2f$" % (kurt_raw,),
                    r"$Kurt_{\mathrm{clean}}=%.2f$" % (kurt_trim,),
                    r"$Outliers=%d$" % (num_outliers,),
                )
            )
            props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgrey")
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=props)
            if s_idx == 0 and a_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_imu_histograms_{rate_tag}.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Saved: {fig_path}")
    plt.close(fig)


def analyze_and_plot_psd(acc, gyr, mag, fs, rate_tag, output_dir, imu_name, prefix):
    """
    Computes and plots Power Spectral Density (PSD) using Welch's method.
    Estimates the -3dB filter cutoff frequency using robust anchored sub-bin interpolation.
    """
    apply_plot_style()
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
    fig.suptitle(f"{imu_name} Sensor Power Spectral Density (Welch's Method) & Interpretation", fontweight="bold")

    nyquist_freq = fs / 2
    sensor_names = ["Accelerometer", "Gyroscope", "Magnetometer"]
    sensor_data = [acc, gyr, mag]
    sensor_units = ["m/s²", "rad/s", "µT"]
    axes_names = ["X", "Y", "Z"]

    for s_idx, (name, data, unit) in enumerate(zip(sensor_names, sensor_data, sensor_units)):
        ax = axs[s_idx]
        for a_idx, ax_name in enumerate(axes_names):
            sig = data[:, a_idx]

            # Compute raw PSD
            f, Pxx = scipy.signal.welch(sig, fs=fs, nperseg=min(len(sig) // 2, 1024))

            # Smooth PSD to find cutoff robustly
            window_len = 11
            cutoff_freq = None
            if len(Pxx) > window_len:
                Pxx_smoothed = np.convolve(Pxx, np.ones(window_len) / window_len, mode="same")

                # Anchor to low-frequency window (1 to 5 Hz) below filter cutoff and above 1/f drift
                low_mask = (f >= 1.0) & (f <= 5.0)
                passband_power = np.median(Pxx_smoothed[low_mask]) if np.any(low_mask) else Pxx_smoothed[0]

                # -3 dB cutoff: power drops below 50% of passband power
                target_val = 0.5 * passband_power
                above_mask = Pxx_smoothed >= target_val
                if np.any(above_mask):
                    last_above_idx = np.max(np.where(above_mask)[0])
                    # Interpolate sub-bin cutoff, ensuring it is below 95% of Nyquist limit
                    if last_above_idx < len(f) - 1 and f[last_above_idx] < 0.95 * f[-1]:
                        f1, f2 = f[last_above_idx], f[last_above_idx + 1]
                        p1, p2 = Pxx_smoothed[last_above_idx], Pxx_smoothed[last_above_idx + 1]
                        cutoff_freq = f1 + (target_val - p1) * (f2 - f1) / (p2 - p1)

            if cutoff_freq is not None:
                ax.loglog(f, Pxx, color=AXIS_COLORS[a_idx], label=f"{ax_name}-axis (-3dB: {cutoff_freq:.1f} Hz)", lw=1.2)
                ax.axvline(cutoff_freq, color=AXIS_COLORS[a_idx], linestyle="--", alpha=0.7, lw=1.0)
            else:
                ax.loglog(f, Pxx, color=AXIS_COLORS[a_idx], label=f"{ax_name}-axis", lw=1.2)

        ax.set_title(f"{name} Power Spectral Density")
        ax.set_ylabel(f"PSD ({unit}² / Hz)")
        ax.set_xlim(0.1, nyquist_freq)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(loc="upper right", ncol=3)

    # Subplot 3: Explanatory / Interpretation Guide
    ax_exp_psd = axs[3]
    f_ideal = np.logspace(-1, np.log10(nyquist_freq), 300)
    f_vibe = 0.2 * nyquist_freq
    f_lpf = 0.4 * nyquist_freq

    Pxx_white = np.ones_like(f_ideal) * 1e-4
    Pxx_drift = 1e-5 / f_ideal
    Pxx_vibe = 1e-3 * np.exp(-0.5 * ((f_ideal - f_vibe) / (0.075 * f_vibe)) ** 2)
    Pxx_alias = 5e-6 * (f_ideal / (0.1 * nyquist_freq)) ** 2

    filter_gain = 1.0 / (1.0 + (f_ideal / f_lpf) ** 3)
    Pxx_total = (Pxx_white + Pxx_drift + Pxx_vibe) * filter_gain + Pxx_alias

    ax_exp_psd.loglog(f_ideal, Pxx_white, color="#76B7B2", linestyle=":", alpha=0.6, label="White Noise Floor")
    ax_exp_psd.loglog(f_ideal, Pxx_drift, color="#E15759", linestyle=":", alpha=0.6, label="1/f Drift Noise")
    ax_exp_psd.loglog(f_ideal, Pxx_vibe, color="#F28E2B", linestyle=":", alpha=0.6, label="Periodic Vibration")
    ax_exp_psd.loglog(f_ideal, Pxx_alias, color="#B07AA1", linestyle=":", alpha=0.6, label="Aliased Noise (Upward curve)")
    ax_exp_psd.loglog(f_ideal, Pxx_total, color="black", lw=2, label="Combined Sensor PSD")

    # Annotate PSD components
    ax_exp_psd.annotate(
        "White Noise Floor\n(Flat region: equal power across freq)",
        xy=(0.02 * nyquist_freq, 1.1e-4),
        xytext=(0.02 * nyquist_freq, 1e-5),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8.5,
    )
    ax_exp_psd.annotate(
        "Periodic Vibration / Hum\n(Sharp peak at resonance freq)",
        xy=(f_vibe, 8e-4),
        xytext=(0.03 * nyquist_freq, 2e-3),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8.5,
    )
    ax_exp_psd.annotate(
        "Bias Drift / Flicker Noise\n(Rising power at low frequency)",
        xy=(0.15, 6e-5),
        xytext=(0.3, 5e-4),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8.5,
    )
    ax_exp_psd.annotate(
        "LPF Roll-off",
        xy=(1.05 * f_lpf, 1.5e-5),
        xytext=(0.6 * f_lpf, 1.5e-6),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8.5,
    )
    ax_exp_psd.annotate(
        "Aliasing Tail (Upward curve)\n(High-frequency fold-back)",
        xy=(0.9 * nyquist_freq, 4.5e-4),
        xytext=(0.06 * nyquist_freq, 5e-5),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8.5,
    )

    text_rule = "DESIGN RULE:\nEnsure hardware IMU filter cut-off\nfrequency ≤ 1/2 f_sample (Nyquist limit)\nto prevent spectral aliasing!"
    props_rule = dict(boxstyle="round", facecolor="#FFF2CC", alpha=0.9, edgecolor="#D6B656")
    ax_exp_psd.text(
        0.02,
        0.05,
        text_rule,
        transform=ax_exp_psd.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        fontweight="bold",
        bbox=props_rule,
    )

    ax_exp_psd.set_title("Explanatory PSD Guide (Idealized)")
    ax_exp_psd.set_ylabel("Power Density (Unit² / Hz)")
    ax_exp_psd.set_xlabel("Frequency (Hz)")
    ax_exp_psd.set_xlim(0.1, nyquist_freq)
    ax_exp_psd.grid(True, which="both", ls="-", alpha=0.2)
    ax_exp_psd.set_ylim(1e-7, 1e-2)
    ax_exp_psd.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_imu_psd_{rate_tag}.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Saved: {fig_path}")
    plt.close(fig)


def analyze_and_plot_allan_deviation(acc, gyr, fs, rate_tag, output_dir, imu_name, prefix):
    """
    Computes overlapping Allan Deviation and plots results for Accelerometer and Gyroscope.
    Overlays slope guides and extracts Random Walk density estimates at tau = 1.0s.
    """
    apply_plot_style()
    print("Calculating overlapping Allan deviation...")

    # Define internal ADEV computation algorithm
    def run_adev_calc(data, sample_fs):
        N = len(data)
        dt = 1.0 / sample_fs
        if data.ndim == 1:
            data = data[:, np.newaxis]
        num_cols = data.shape[1]

        max_m = N // 10
        if max_m < 1:
            max_m = 1
        m_values = np.unique(np.geomspace(1, max_m, num=100, dtype=int))

        taus = m_values * dt
        adevs = np.zeros((len(m_values), num_cols))
        cum_data = np.vstack((np.zeros((1, num_cols)), np.cumsum(data, axis=0)))

        for idx, m in enumerate(m_values):
            sum_clusters = cum_data[m:] - cum_data[:-m]
            means = sum_clusters / m
            diff = means[m:] - means[:-m]
            variance = np.mean(diff**2, axis=0) / 2.0
            adevs[idx] = np.sqrt(variance)

        return taus, adevs

    allan_start = time.perf_counter()
    taus_acc, adevs_acc = run_adev_calc(acc, fs)
    taus_gyr, adevs_gyr = run_adev_calc(gyr, fs)
    print(f"Allan deviation computed in {time.perf_counter() - allan_start:.3f} seconds.")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"{imu_name} Overlapping Allan Deviation Analysis & Interpretation", fontweight="bold")

    axes_names = ["X", "Y", "Z"]

    # 1. Accelerometer ADEV
    ax_acc = axs[0]
    for a_idx, ax_name in enumerate(axes_names):
        ax_acc.loglog(taus_acc, adevs_acc[:, a_idx], color=AXIS_COLORS[a_idx], label=f"{ax_name}-axis", lw=1.8)
    c_acc = adevs_acc[0, 0] * (taus_acc[0] ** 0.5)
    ax_acc.loglog(
        taus_acc, c_acc * (taus_acc**-0.5), color="gray", linestyle=":", label="Slope -0.5 (White Noise guide)", alpha=0.8
    )
    ax_acc.axvline(1.0, color="dimgray", linestyle="--", alpha=0.5)
    idx_1s_acc = np.argmin(np.abs(taus_acc - 1.0))
    annotation_text_acc = "VRW (Noise Density) at τ=1s:\n"
    for a_idx, ax_name in enumerate(axes_names):
        val = adevs_acc[idx_1s_acc, a_idx]
        ax_acc.plot(taus_acc[idx_1s_acc], val, "o", color=AXIS_COLORS[a_idx], markersize=6)
        annotation_text_acc += f"  {ax_name}-axis: {val:.5f} (m/s²)/√Hz\n"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgrey")
    ax_acc.text(
        0.05,
        0.05,
        annotation_text_acc.strip(),
        transform=ax_acc.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=props,
    )
    ax_acc.set_title("Accelerometer Allan Deviation")
    ax_acc.set_xlabel("Correlation Time τ (s)")
    ax_acc.set_ylabel("Allan Deviation σ(τ) (m/s²)")
    ax_acc.grid(True, which="both", ls="-", alpha=0.2)
    ax_acc.legend(loc="upper right", fontsize=8)

    # 2. Gyroscope ADEV
    ax_gyr = axs[1]
    for a_idx, ax_name in enumerate(axes_names):
        ax_gyr.loglog(taus_gyr, adevs_gyr[:, a_idx], color=AXIS_COLORS[a_idx], label=f"{ax_name}-axis", lw=1.8)
    c_gyr = adevs_gyr[0, 0] * (taus_gyr[0] ** 0.5)
    ax_gyr.loglog(
        taus_gyr, c_gyr * (taus_gyr**-0.5), color="gray", linestyle=":", label="Slope -0.5 (White Noise guide)", alpha=0.8
    )
    ax_gyr.axvline(1.0, color="dimgray", linestyle="--", alpha=0.5)
    idx_1s_gyr = np.argmin(np.abs(taus_gyr - 1.0))
    annotation_text_gyr = "ARW (Noise Density) at τ=1s:\n"
    for a_idx, ax_name in enumerate(axes_names):
        val = adevs_gyr[idx_1s_gyr, a_idx]
        ax_gyr.plot(taus_gyr[idx_1s_gyr], val, "o", color=AXIS_COLORS[a_idx], markersize=6)
        annotation_text_gyr += f"  {ax_name}-axis: {val:.5f} (rad/s)/√Hz\n"
    ax_gyr.text(
        0.05,
        0.05,
        annotation_text_gyr.strip(),
        transform=ax_gyr.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=props,
    )
    ax_gyr.set_title("Gyroscope Allan Deviation")
    ax_gyr.set_xlabel("Correlation Time τ (s)")
    ax_gyr.set_ylabel("Allan Deviation σ(τ) (rad/s)")
    ax_gyr.grid(True, which="both", ls="-", alpha=0.2)
    ax_gyr.legend(loc="upper right", fontsize=8)

    # 3. Explanatory ADEV Guide
    ax_exp = axs[2]
    tau_ideal = np.logspace(-3, 4, 300)
    Q, N_val, B_val, K_val, R_val = 2e-3, 1e-3, 1e-4, 1e-5, 2e-6
    adev_qn = np.sqrt(3 * Q**2 / tau_ideal**2)
    adev_wn = np.sqrt(N_val**2 / tau_ideal)
    adev_bi = np.ones_like(tau_ideal) * B_val
    adev_rw = np.sqrt(K_val**2 * tau_ideal / 3)
    adev_br = np.sqrt(R_val**2 * tau_ideal**2 / 2)
    adev_total = np.sqrt(
        3 * Q**2 / tau_ideal**2 + N_val**2 / tau_ideal + B_val**2 + K_val**2 * tau_ideal / 3 + R_val**2 * tau_ideal**2 / 2
    )

    ax_exp.loglog(tau_ideal, adev_qn, color="#E15759", linestyle=":", alpha=0.6, label="Quantization Noise (Slope -1.0)")
    ax_exp.loglog(
        tau_ideal, adev_wn, color="#76B7B2", linestyle=":", alpha=0.6, label="White Noise / Random Walk (Slope -0.5)"
    )
    ax_exp.loglog(tau_ideal, adev_bi, color="#4E79A7", linestyle=":", alpha=0.6, label="Bias Instability (Slope 0.0)")
    ax_exp.loglog(tau_ideal, adev_rw, color="#F28E2B", linestyle=":", alpha=0.6, label="Rate Random Walk (Slope +0.5)")
    ax_exp.loglog(tau_ideal, adev_br, color="#B07AA1", linestyle=":", alpha=0.6, label="Bias Ramp (Slope +1.0)")
    ax_exp.loglog(tau_ideal, adev_total, color="black", lw=2.5, label="Combined IMU ADEV")

    # Annotate ADEV properties
    ax_exp.plot(1.0, N_val, "ko", markersize=6)
    ax_exp.axvline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax_exp.annotate(
        "ARW / VRW\nValue at τ = 1s",
        xy=(1.0, N_val),
        xytext=(1.5, N_val * 2),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8,
    )

    min_idx = np.argmin(adev_total)
    tau_min = tau_ideal[min_idx]
    val_min = adev_total[min_idx]
    ax_exp.plot(tau_min, val_min, "ko", markersize=6)
    ax_exp.annotate(
        "Bias Instability Limit\nValue at Minimum",
        xy=(tau_min, val_min),
        xytext=(tau_min * 2, val_min * 0.3),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8,
    )

    # Annotate slope regions
    ax_exp.text(1.5e-3, 3e-1, "Quantization\n(Slope -1)", fontsize=8, color="#E15759", fontweight="bold")
    ax_exp.text(1.5e-2, 3e-3, "White Noise\n(Slope -0.5)", fontsize=8, color="#76B7B2", fontweight="bold")
    ax_exp.text(8.0, 3.5e-5, "Bias Instability\n(Slope 0)", fontsize=8, color="#4E79A7", fontweight="bold")
    ax_exp.text(2e2, 2.5e-4, "Random Walk\n(Slope +0.5)", fontsize=8, color="#F28E2B", fontweight="bold")
    ax_exp.text(1.5e3, 5e-2, "Ramp\n(Slope +1)", fontsize=8, color="#B07AA1", fontweight="bold")

    ax_exp.set_title("Explanatory ADEV Guide (Idealized)")
    ax_exp.set_xlabel("Correlation Time τ (s)")
    ax_exp.set_ylabel("Allan Deviation σ(τ)")
    ax_exp.set_ylim(1e-5, 1e1)
    ax_exp.grid(True, which="both", ls="-", alpha=0.2)
    ax_exp.legend(loc="upper right", fontsize=7.5, ncol=1)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_imu_allan_deviation_{rate_tag}.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Saved: {fig_path}")
    plt.close(fig)


# --- MAIN RUNNER BLOCK ---


def main():
    # Clear terminal
    print("\033[H\033[2J")
    print("=== IMU Sensor Characterization and Noise Analysis ===\n")

    # Determine default parameters based on constant selection
    is_yost = IMU_TYPE == "tss"
    imu_name = "Yost Labs 3-Space" if is_yost else "WitMotion"
    prefix = "tss" if is_yost else "wit"

    # 1. Recording Live Data (figures are written alongside the recording they came from)
    if RECORD_DATA:
        save_dir = make_run_dir("imu_char")
        data_path = record_data(imu_type=IMU_TYPE, duration=DURATION, save_dir=save_dir)
    else:
        if not DATA_FILE:
            print("Error: RECORD_DATA is False but DATA_FILE is empty.")
            print("Set DATA_FILE to an existing .npz recording or set RECORD_DATA = True to record from a live sensor.")
            sys.exit(1)
        data_path = pathlib.Path(DATA_FILE).expanduser().resolve()
        save_dir = data_path.parent

    # 2. Loading Saved Data File
    try:
        t, acc, gyr, mag, columns = load_data(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that your DATA_FILE path is correct or set RECORD_DATA = True to record from a live sensor.")
        sys.exit(1)

    # 3. Sampling Statistics and Jitter
    dt_intervals = np.diff(t)
    mean_fs = 1.0 / np.mean(dt_intervals)
    f_sample_t = 1.0 / dt_intervals
    std_fs = np.std(f_sample_t)
    rate_tag = f"{int(round(mean_fs))}Hz"

    print("\n=== Sampling Performance ===")
    print(f"Average sample rate: {mean_fs:.2f} Hz")
    print(f"Sampling interval dt: {np.mean(dt_intervals) * 1000:.3f} ± {np.std(dt_intervals) * 1000:.3f} ms")
    print(f"Sample rate jitter (std dev): {std_fs:.3f} Hz ({(std_fs / mean_fs) * 100:.2f}% of mean)")

    # 4. Standard and Trimmed Statistics
    print_sensor_stats("Accelerometer", acc, "m/s²")
    print_sensor_stats("Gyroscope", gyr, "rad/s")
    print_sensor_stats("Magnetometer", mag, "µT")

    # Vector norm stability analysis
    mag_norm = np.linalg.norm(mag, axis=1)
    print("\n--- Magnetometer Vector Norm Analysis ---")
    print(f"Mean field intensity: {np.mean(mag_norm):.5f} µT")
    print(f"Standard deviation: {np.std(mag_norm):.5f} µT ({(np.std(mag_norm) / np.mean(mag_norm)) * 100:.3f}% variation)")

    # 5. Visualizing performance
    print("\n=== Generating and Saving Figures ===")
    plot_time_series(t, acc, gyr, mag, rate_tag, save_dir, imu_name, prefix)
    plot_noise_histograms(acc, gyr, mag, rate_tag, save_dir, imu_name, prefix)
    analyze_and_plot_psd(acc, gyr, mag, mean_fs, rate_tag, save_dir, imu_name, prefix)
    analyze_and_plot_allan_deviation(acc, gyr, mean_fs, rate_tag, save_dir, imu_name, prefix)

    print(f"\nAnalysis and visualization complete. All plots saved to '{save_dir}/'.")


if __name__ == "__main__":
    main()
