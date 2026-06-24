"""Controller parameters, ported from baseline_controller/params.m and its .mat
files (sys_id_*, ankle_stiffness_params, spring_zero_positions, encoder_counts_
to_rad_polys, ankle_angle_offset, dynamic_params).

Needs cleanup
"""
from dataclasses import dataclass
from math import pi


# Common / shared
@dataclass(frozen=True, slots=True)
class CommonParams:
    sample_rate: float = 1000.0              # Hz
    sample_time: float = 0.001               # s

    # ticks after control_enabled rises during which the pipeline runs but drives
    # are muted to zero (filters settle); Simulink LLC "1 s Delay" = 1000 @ 1 kHz
    startup_mute_ticks: int = 1000

    gravity: float = 9.81                    # m/s^2
    swing_body_weight: float = 80.0          # kg
    stance_body_weight: float = 75.0         # kg

    torque_rate_lim: float = 20.0            # Nm/s
    damping_gain: float = 0.15               # outer damping comp (both joints)

    # default torque-conditioning lowpass; unused until those blocks take params
    lowpass_cutoff: float = 25.0             # Hz
    lowpass_order: int = 2


# Knee
@dataclass(frozen=True, slots=True)
class KneeParams:
    # motor / drivetrain
    torque_constant: float                   # Nm/A
    rated_current: float                     # A
    gear_ratio: float
    vel_max: float                           # rad/s, joint side
    motor_damping: float
    motor_inertia: float
    motor_eff: float
    motor_abs_counts_to_rad: float           # 2*pi / 2^30

    stiffness: float                         # Nm/rad

    kp: float
    wc_ff: float                             # rad/s, Qff cutoff
    wc_dob: float                            # rad/s, DOB cutoff

    angle_min: float
    angle_min_soft: float
    angle_max: float
    angle_max_soft: float
    torque_max: float                        # Nm
    c_jointstop: float                       # Nm/rad, soft-stop stiffness
    w_max_jointstop: float                   # rad/s, soft-stop retract speed

    # per-stop sign multipliers (+1 upper, -1 lower). Wiring into
    # KneeTorqueConditioning pending. See [[low-level-conditioning-ext-pending]].
    flex_ext: float                          # at angle_max_soft (+1)
    extend_ext: float                        # at angle_min_soft (-1)

    vel_sat_lowpass_cutoff: float            # Hz
    vel_sat_lowpass_order: int

    @property
    def up_limit(self) -> float:
        """Alias for the conditioning block's `knee_up_limit`."""
        return self.angle_max_soft


# Ankle
@dataclass(frozen=True, slots=True)
class AnkleParams:
    # motor / drivetrain
    torque_constant: float                   # Nm/A
    rated_current: float                     # A
    gear_ratio: float
    vel_max: float                           # rad/s, joint side
    motor_damping: float
    motor_inertia: float
    motor_eff: float                         # efficiency comp disabled (arch.md)
    motor_abs_counts_to_rad: float           # 2*pi / 2^30

    stiffness: float                         # Nm/rad, linear
    stiffness_poly: tuple[float, float]      # Nm/rad nonlinear fit, not yet used
    angle_offset: float                      # rad

    kp: float

    angle_min: float
    angle_min_soft: float
    angle_max: float
    angle_max_soft: float
    torque_max: float                        # Nm
    c_jointstop: float                       # Nm/rad
    w_max_jointstop: float                   # rad/s

    # per-stop sign multipliers (see KneeParams)
    flex_ext: float
    extend_ext: float

    vel_sat_lowpass_cutoff: float            # Hz
    vel_sat_lowpass_order: int

    # ankle torque controller spring-velocity lowpass (spring-accel branch unported)
    torque_ctrl_spring_vel_lowpass_cutoff: float   # Hz
    torque_ctrl_spring_vel_lowpass_order: int


# Encoders (spring_zero_positions.mat) — linear scales, zeros, hall offsets.
# Motor counts-to-rad lives on the joint params. Used by Get Prosthesis State.
@dataclass(frozen=True, slots=True)
class EncoderParams:
    knee_abs_counts_to_rad: float            # high-res abs joint encoder (empirical, sign-flipped)
    ankle_abs_counts_to_rad: float           # 2*pi / 2^17 (17-bit abs)

    knee_abs_counts_offset: int              # knee0pos
    ankle_abs_counts_offset: int             # ankle0pos
    knee_motor_abs_counts_offset: int        # kneemotor0pos
    ankle_motor_abs_counts_offset: int       # anklemotor0pos

    hall_0_offset: int
    hall_1_offset: int


# IMU (two WitMotion HWT906, thigh + foot, over USB-serial). Set the port paths
# from imu_io.discover_imu_ports() (which CH340 is which IMU) — they can swap across
# reboots/replugs, so update here after discovery. acc/gyro ranges match the
# baseline Simulink Read IMU block (±16 g, ±2000 dps).
@dataclass(frozen=True, slots=True)
class IMUParams:
    thigh_port: str            # set from discover_imu_ports()
    foot_port: str
    baud: int                  # 921600 reliable on Linux
    stream_rate_hz: int        # sensor output rate
    lpf_cutoff_hz: int         # anti-alias; keep <= 0.4 * stream_rate
    acc_range_g: int           # ±g
    gyro_range_dps: int        # ±dps


# Encoder error correction (encoder_counts_to_rad_polys.mat). Knee angle and
# ankle spring angle each add a normalized polynomial correction: normalize
# counts ((x-mean)/std), eval *_err_poly, de-normalize (*err_std + err_mean).
# Polys are highest-power-first (polyval order).
@dataclass(frozen=True, slots=True)
class EncoderCorrection:
    knee_counts_mean: float
    knee_counts_std: float
    knee_err_poly: tuple[float, ...]
    knee_err_mean: float
    knee_err_std: float

    ankle_spring_counts_to_rad_poly: tuple[float, ...]
    ankle_spring_counts_mean: float
    ankle_spring_counts_std: float
    ankle_spring_err_poly: tuple[float, ...]
    ankle_spring_err_mean: float
    ankle_spring_err_std: float

    ankle_total_counts: int                  # 17-bit wrap modulus


# Standing (baseline "Standing Mode Control"). Min-jerk ramp from the latched
# start pose to a fixed pose over `duration`, then per-joint PD:
#   tau = kp*(theta_traj - theta) - kd*theta_dot_filt
# theta_dot_filt = joint velocity through a Butterworth lowpass (angle stays raw).
@dataclass(frozen=True, slots=True)
class StandingParams:
    knee_target: float                       # rad
    ankle_target: float                      # rad
    duration: float                          # s, min-jerk ramp
    kp: float                                # Nm/rad
    kd: float                                # Nm/(rad/s)

    knee_vel_lowpass_cutoff: float           # Hz, F3dB
    ankle_vel_lowpass_cutoff: float          # Hz, F3dB
    vel_lowpass_order: int                   # Butterworth order


# Default instances (loaded once at import). Inject alternates in tests.
COMMON = CommonParams()

KNEE = KneeParams(
    torque_constant=0.065,
    rated_current=20.0,
    gear_ratio=50.0,
    vel_max=5000.0 * 2.0 * pi / 60.0 / 50.0 * 0.95,
    motor_damping=9.7032,
    motor_inertia=0.5340,
    motor_eff=0.9129,
    motor_abs_counts_to_rad=2.0 * pi / (2 ** 30),
    stiffness=1666.0,
    kp=0.09,
    wc_ff=25.0 * 2.0 * pi,
    wc_dob=10.0 * 2.0 * pi,
    angle_min=-2.0 * pi / 180.0,
    angle_min_soft=5.0 * pi / 180.0,
    angle_max=90.0 * pi / 180.0,
    angle_max_soft=85.0 * pi / 180.0,
    torque_max=100.0,
    c_jointstop=160.0,
    w_max_jointstop=10.0,
    flex_ext=+1.0,
    extend_ext=-1.0,
    vel_sat_lowpass_cutoff=75.0,
    vel_sat_lowpass_order=2,
)

ANKLE = AnkleParams(
    torque_constant=0.0529,
    rated_current=15.7,
    gear_ratio=100.0,
    vel_max=7000.0 * 2.0 * pi / 60.0 / 100.0 * 0.95,
    motor_damping=5.2961,
    motor_inertia=0.4683,
    motor_eff=0.5583,
    motor_abs_counts_to_rad=2.0 * pi / (2 ** 30),
    stiffness=993.8613,
    stiffness_poly=(1080.4, -6216.5),
    angle_offset=0.0444,
    kp=0.08,
    angle_min=-30.0 * pi / 180.0,
    angle_min_soft=-20.0 * pi / 180.0,
    angle_max=30.0 * pi / 180.0,
    angle_max_soft=20.0 * pi / 180.0,
    torque_max=150.0,
    c_jointstop=80.0,
    w_max_jointstop=5.0,
    flex_ext=+1.0,
    extend_ext=-1.0,
    vel_sat_lowpass_cutoff=50.0,
    vel_sat_lowpass_order=2,
    torque_ctrl_spring_vel_lowpass_cutoff=20.0,
    torque_ctrl_spring_vel_lowpass_order=2,
)

ENCODERS = EncoderParams(
    knee_abs_counts_to_rad=-1.481916095218253e-08,   # verified (read_encoders); NOT 2pi/2^17
    ankle_abs_counts_to_rad=2.0 * pi / (2 ** 17),
    knee_abs_counts_offset=234_149_441,
    ankle_abs_counts_offset=-553,
    knee_motor_abs_counts_offset=-2_635,
    ankle_motor_abs_counts_offset=-951,
    hall_0_offset=1_738,
    hall_1_offset=2_261,
)

IMU = IMUParams(
    thigh_port="/dev/ttyUSB1",   # update from discover_imu_ports()
    foot_port="/dev/ttyUSB0",
    baud=921600,
    stream_rate_hz=200,        # tune later; 0.4*200=80 -> lpf 42 is the largest valid
    lpf_cutoff_hz=42,
    acc_range_g=16,
    gyro_range_dps=2000,
)

ENCODER_CORRECTION = EncoderCorrection(
    knee_counts_mean=-49_248_866.40327176,
    knee_counts_std=33_291_983.071372624,
    knee_err_poly=(0.47774077, -0.38791886, -0.80362217, 0.50741695),
    knee_err_mean=-6.0769758202966855e-06,
    knee_err_std=0.00037360483505769084,
    ankle_spring_counts_to_rad_poly=(-1.27664165e-15, 2.81074451e-10, 4.75918169e-05, 0.0),
    ankle_spring_counts_mean=-1248.7238249778297,
    ankle_spring_counts_std=7852.009105751588,
    ankle_spring_err_poly=(2.37075304, -0.26480173, -4.05764862, -0.08075658, 0.56996956),
    ankle_spring_err_mean=-0.00038694562088488905,
    ankle_spring_err_std=0.0008505143990917394,
    ankle_total_counts=2 ** 17,
)

STANDING = StandingParams(
    knee_target=5.0 * pi / 180.0,            # 5 deg
    ankle_target=0.0,
    duration=2.0,
    kp=150.0,
    kd=5.0,
    knee_vel_lowpass_cutoff=25.0,
    ankle_vel_lowpass_cutoff=20.0,
    vel_lowpass_order=2,
)

# dynamic_params.mat -> dynpars (9x1), for the high-level controller (not yet
# ported). MATLAB order preserved; semantic mapping unconfirmed — label before use.
DYNPARS: tuple[float, ...] = (
    0.0742, 1.5952, 0.0064, 3.5057, 0.0896,
    2.5055, 0.0305, 0.0117, 0.0865,
)
