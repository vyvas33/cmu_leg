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

    # Measured-dt clamp, applied wherever a real elapsed dt is divided by.
    # The loop schedules 1 kHz but does not always make it: logged standing runs
    # sit at 1.000 ms median / 1.02 ms p99, with a ~21 ms outlier once per 15 s
    # run and ~0.7% of ticks under 0.5 ms (the catch-up burst after an overrun).
    # dt_max keeps the 21 ms tick from being integrated as one step; dt_min
    # bounds quantization-noise gain on the differentiators -- velocity noise
    # goes as 1/dt, and a 17-bit encoder count (4.8e-5 rad) over 0.19 ms is
    # already 0.25 rad/s of pure noise.
    dt_min: float = 0.0005                   # s
    dt_max: float = 0.005                    # s

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


@dataclass(frozen=True, slots=True)
class ImpedanceStanceGains:
    knee_k: float #Nm/rad
    knee_b: float #Nm/(rad/s) 
    knee_off: float #rad
    ankle_k: float 
    ankle_b: float 
    ankle_off: float

@dataclass(frozen=True, slots=True)
class ImpedanceParams:
    stance_12_threshold: float
    stance_23_threshold: float 
    stance1: ImpedanceStanceGains
    stance2: ImpedanceStanceGains
    stance3: ImpedanceStanceGains



# Default instances (loaded once at import). Inject alternates in tests.
COMMON = CommonParams()

KNEE = KneeParams(
    torque_constant=0.065,
    rated_current=20.0,
    gear_ratio=50.0,
    vel_max=2000 * 2.0 * pi / 60.0 / 50.0 * 0.95, #was 5000
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
    # 0 to match NPC params.m (ankle_angle_offset.mat load is commented out
    # there; the Sep-2025 zeroing assumes no extra offset). Old value: 0.0444.
    angle_offset=0.0,
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

# updated july 2026
ENCODERS = EncoderParams(
    knee_abs_counts_to_rad=-1.4817901340709475e-08,  # empirical; NOT 2pi/2^17
    ankle_abs_counts_to_rad=2.0 * pi / (2 ** 17),
    knee_abs_counts_offset=234_361_752,
    ankle_abs_counts_offset=-428,
    knee_motor_abs_counts_offset=-54,
    ankle_motor_abs_counts_offset=-2_328,
    hall_0_offset=1_786,
    hall_1_offset=2270,
)

IMU = IMUParams(
    thigh_port="/dev/ttyUSB1",   # update from discover_imu_ports()
    foot_port="/dev/ttyUSB0",
    baud=921600,
    stream_rate_hz=1000,        # tune later; 0.4*200=80 -> lpf 42 is the largest valid
    lpf_cutoff_hz=256,
    acc_range_g=16,
    gyro_range_dps=2000,
)

# From prosthesis_control_NPC/utility_functions/encoder_counts_to_rad_polys.mat
# (Feb 2024) — the set paired with the Sep-2025 zero offsets above. Polys,
# means and stds must always come from the same .mat as each other.
ENCODER_CORRECTION = EncoderCorrection(
    knee_counts_mean=-45_904_738.257438816,
    knee_counts_std=37_310_443.63765212,
    knee_err_poly=(0.05548087522230831, -1.1104694633610175,
                   0.09695060250841925, 0.9138964820829386),
    knee_err_mean=0.00020196764386727568,
    knee_err_std=0.0003709472820373927,
    ankle_spring_counts_to_rad_poly=(3.6598573670211066e-15, 2.4092807648254984e-10,
                                     4.738107395479765e-05, 0.0),
    ankle_spring_counts_mean=91.98335817656937,
    ankle_spring_counts_std=5797.450564709744,
    ankle_spring_err_poly=(2.0472565773960527, -0.6925316761570832,
                           -3.369650930587924, 0.593056076788454,
                           0.6740383730007847),
    ankle_spring_err_mean=-0.0004594550864598505,
    ankle_spring_err_std=0.0006921761420068215,
    ankle_total_counts=2 ** 17,
)

STANDING = StandingParams(
    knee_target=5.0 * pi / 180.0,            # 5 deg
    ankle_target=0.0,
    duration=2.0,
    kp=150.0,
    kd=0.0, #5.0,
    knee_vel_lowpass_cutoff=25.0,
    ankle_vel_lowpass_cutoff=20.0,
    vel_lowpass_order=2,
)

# 3-state finite-state impedance stance (ImpedanceStance). Each phase applies
#   tau = -k*(angle - off) - b*angle_dot   for knee and ankle.
# Phase advance: Stance1 -> Stance2 when knee_angle > stance_12_threshold,
#                Stance2 -> Stance3 when ankle_angle > stance_23_threshold.
# Reference layout: impedance_control_basic/ParameterLibrary/impedance_control/
#   optimized_gains_linear.mat (20-vector). k [Nm/rad], b [Nm/(rad/s), floor 0.02],
#   off/thresholds [rad]. PLACEHOLDERS — tune before enabling.
IMPEDANCE = ImpedanceParams(
    stance_12_threshold=0.0,   # rad, on knee_angle
    stance_23_threshold=0.0,   # rad, on ankle_angle
    stance1=ImpedanceStanceGains(
        knee_k=0.0,  knee_b=0.0,  knee_off=0.0,
        ankle_k=0.0, ankle_b=0.0, ankle_off=0.0,
    ),
    stance2=ImpedanceStanceGains(
        knee_k=0.0,  knee_b=0.0,  knee_off=0.0,
        ankle_k=0.0, ankle_b=0.0, ankle_off=0.0,
    ),
    stance3=ImpedanceStanceGains(
        knee_k=0.0,  knee_b=0.0,  knee_off=0.0,
        ankle_k=0.0, ankle_b=0.0, ankle_off=0.0,
    ),
)

# dynamic_params.mat -> dynpars (9x1), for the high-level controller (not yet
# ported). MATLAB order preserved; semantic mapping unconfirmed — label before use.
DYNPARS: tuple[float, ...] = (
    0.0742, 1.5952, 0.0064, 3.5057, 0.0896,
    2.5055, 0.0305, 0.0117, 0.0865,
)


# ESKF (estimation/eskf). PLACEHOLDER values pending tuning on hardware.
# Q sigmas follow the baseline convention: each 30-dim Q diagonal entry is
# (sigma * dt)^2, per-body order [dtheta, gyro bias, pos (0), vel, accel bias].
# sigma_dtheta/sigma_dv drive the attitude/velocity random walks (IMU white
# noise); sigma_gyr/sigma_acc drive the bias random walks. Values from
# params_new.m (the set nm_nn.m actually loads; params.m has an older set).
@dataclass(frozen=True, slots=True)
class ESKFConfig:
    dt: float                    # nominal filter tick [s]; fallback when dt is not measurable
    # measured-dt clamp, as multiples of the nominal dt. The filter integrates
    # the caller's clock delta (t - t_prev), so loop jitter is absorbed; the
    # clamp keeps a scheduler stall or a clock glitch from injecting a huge step.
    # DEAD as of the measured-dt refactor: the ESKF no longer derives its own
    # dt, the caller passes one already clamped to COMMON.dt_min/dt_max. Kept
    # only so the ESKFConfig literal below still constructs; delete both once
    # nothing references them.
    dt_min_factor: float
    dt_max_factor: float
    # an IMU sample older than this (host clock, same timebase as t) means the
    # USB stream stalled -- zero-order hold would silently integrate frozen
    # data, so the filter reports itself unhealthy instead.
    imu_stale_timeout: float     # [s]
    # process noise sigmas, thigh
    sigma_dtheta_thigh: float    # attitude random walk [rad/s]
    sigma_gyr_thigh: float       # gyro-bias walk [rad/s^2... baseline units, unverified]
    sigma_dv_thigh: float        # velocity random walk [m/s^2]
    sigma_acc_thigh: float       # accel-bias walk
    # process noise sigmas, foot
    sigma_dtheta_foot: float
    sigma_gyr_foot: float
    sigma_dv_foot: float
    sigma_acc_foot: float
    # measurement noise
    sigma_enc: float             # encoder angle noise [rad]
    sigma_rate: float            # joint-rate noise [rad/s]; raw derivative -> expect big
    sigma_model_pos: float       # kinematic-model floor, position closure [m]
    sigma_model_vel: float       # kinematic-model floor, velocity closure [m/s]
    sigma_zupt: float            # zero-velocity pseudo-measurement [m/s]
    # initial error covariance, per-body diag VARIANCES (baseline Delay_P init;
    # both bodies identical). pos = 0: the start pose defines the nav origin.
    p0_att: float                # [rad^2]  ~ comp-filter handoff quality
    p0_gyr_bias: float           # [(rad/s)^2]
    p0_pos: float                # [m^2]
    p0_vel: float                # [(m/s)^2]
    p0_acc_bias: float           # [(m/s^2)^2]
    # comp-filter init
    init_duration: float         # [s] comp filter runs this long before ESKF starts
    comp_beta: float             # comp-filter pull rate [rad/s]; > worst tilt / init_duration
    # inflate Q's dtheta/dv terms with the measured integration error (change in
    # corrected acc/gyro between ticks) instead of white noise alone
    dynamic_process_noise: bool


ESKF_CFG = ESKFConfig(
    dt=0.001,
    dt_min_factor=0.5,
    dt_max_factor=5.0,
    imu_stale_timeout=0.05,   # IMU streams ~200 Hz (5 ms); 50 ms = 10 missed
    sigma_dtheta_thigh=0.01, sigma_gyr_thigh=0.001,
    sigma_dv_thigh=1.0, sigma_acc_thigh=0.001,
    sigma_dtheta_foot=0.01, sigma_gyr_foot=0.001,
    sigma_dv_foot=1.0, sigma_acc_foot=0.001,
    sigma_enc=0.01,
    sigma_rate=0.1,
    sigma_model_pos=0.01,
    sigma_model_vel=0.01,
    sigma_zupt=0.01,
    p0_att=1e-4, p0_gyr_bias=1e-4, p0_pos=0.0, p0_vel=1e-3, p0_acc_bias=1e-3,
    init_duration=1.0,
    comp_beta=5.0,
    dynamic_process_noise=False,
)


# Leg geometry for the ESKF kinematic chain (estimation/eskf). Decoded from the
# baseline ESKF transforms (prosthesis_control_NPC/ESKF/T_O*_func_eskf_new.m)
# evaluated at identity attitude, with lengths from params_new.m.
# Body frame convention (see eskf/kinematics.py): +x forward, +y left, +z up.
@dataclass(frozen=True, slots=True)
class LegGeometry:
    shank_length: float                          # ankle joint -> knee joint [m]
    knee_to_thigh_imu: tuple[float, float, float]  # knee -> thigh IMU origin, thigh frame [m]
    foot_imu_to_ankle: tuple[float, float, float]  # foot IMU origin -> ankle, foot frame [m]
    ankle_to_toe: tuple[float, float, float]     # ankle -> toe, foot frame [m]
    ankle_to_heel: tuple[float, float, float]    # ankle -> heel, foot frame [m]
    hip_to_knee: float                           # SUBJECT SPECIFIC, see note below [m]
    knee_angle_offset: float                     # baseline 'gamma' [rad], see note
    thigh_imu_pitch: float                       # IMU mount tilt [rad], TUNING, see note


LEG = LegGeometry(
    # = params_new.m length_shank; baseline trans_knee_ankle = [0;0;-length_shank]
    # confirms the offset is purely along the link.
    shank_length=0.215,
    # From T_OK at identity: knee sits at [-0.0846, -0.0550, -0.1040] in the
    # thigh IMU frame, so the IMU is that far the other way. NOT a pure offset
    # along the link: the IMU is mounted 5.5 cm off the sagittal plane and
    # 8.5 cm fore-aft, only 10.4 cm above the knee. The baseline's alpha
    # (0.67 deg thigh-IMU pitch mounting offset) is ALREADY baked into these
    # numbers -- do not apply it a second time.
    knee_to_thigh_imu=(0.084567, 0.055, 0.103991),
    # CONFIRMED zero on our hardware (Jul 2026): the foot IMU sits on the ankle
    # joint itself. Matches the baseline, which never models this offset -- its
    # loop closure walks thigh -> knee -> ankle and compares that straight to
    # the foot position.
    foot_imu_to_ankle=(0.0, 0.0, 0.0),
    ankle_to_toe=(0.156, 0.0, -0.096),
    ankle_to_heel=(-0.09, 0.0, -0.096),
    # params_new.m length_thigh. Depends on the wearer (the file lists 0.55 to
    # 0.71 for different subjects); 0.66 is "John". Not used by the loop
    # closures, needed for hip position when leg kinematics is ported.
    hip_to_knee=0.66,
    # baseline gamma = 6 deg, added to the knee angle before the shank
    # rotation. Kept for reference ONLY -- see thigh_imu_pitch below for why we
    # do not apply it as a knee offset. Unused.
    knee_angle_offset=6.0 * pi / 180.0,
    # Fixed pitch of the thigh IMU mount about the body +y (sagittal) axis.
    # POSITIVE = the sensor's +x (forward) axis tilts UP relative to the thigh
    # link, so the sensor->body map is rot_y(-thigh_imu_pitch) @ <axis
    # permutation> (eskf.py) and knee_to_thigh_imu, which is expressed in the
    # SENSOR frame, gets the same rot_y(-pitch) applied (measurements.py).
    # Both effects MUST stay coupled: one physical tilt rotates the frame the
    # chain descends in AND re-expresses the knee->IMU vector. This is the
    # correct home for the constant angular offset -- putting it in THETA0_KNEE
    # instead would get the attitude sum right but leave the knee position
    # wrong by |knee_to_thigh_imu| * pitch (~14 mm at 6 deg).
    #
    # CONFIRMED Jul 2026: the thigh IMU sits 6 deg NEGATIVE about body +y.
    # Note this is OUR mount, established on our hardware -- it is not the
    # baseline's gamma (6 deg at the knee slot), which stays unused; the numeric
    # match is a coincidence of magnitude, not the same physical quantity.
    thigh_imu_pitch=-6.0 * pi / 180.0,
)
