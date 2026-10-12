"""Controller parameters, ported from baseline_controller/params.m and its .mat
files (sys_id_*, ankle_stiffness_params, spring_zero_positions, encoder_counts_
to_rad_polys, ankle_angle_offset, dynamic_params).

Provenance, derivations and the reasoning behind individual values are in
docs/params.md. Keep this file to one line per field and put the prose there.
"""
from dataclasses import dataclass
from math import pi, acos


# Common / shared
@dataclass(frozen=True, slots=True)
class CommonParams:
    sample_rate: float = 1000.0              # Hz, low-level loop rate
    sample_time: float = 0.001               # s
    control_sample_rate: float = 500.0       # Hz, control process (ESKF + high level)

    dt_min: float = 0.0005                   # s, measured-dt clamp, all derivatives
    dt_max: float = 0.005                    # s, stops a stalled tick integrating whole

    startup_mute_ticks: int = 1000           # post-enable ticks run muted

    gravity: float = 9.81                    # m/s^2
    swing_body_weight: float = 80.0          # kg
    stance_body_weight: float = 75.0         # kg

    torque_rate_lim: float = 20.0            # Nm/s
    damping_gain: float = 0.15               # outer damping comp (both joints)

    lowpass_cutoff: float = 25.0             # Hz, torque conditioning; unused so far
    lowpass_order: int = 2


# Multiprocess runtime (runtime/launch.py + low_level.py + control.py).
# Orchestration and safety knobs; loop periods derive from CommonParams.
@dataclass(frozen=True, slots=True)
class RuntimeParams:
    tmax: float = 60.0                  # s, from the FIRST ACTUATING TICK, not t0
    max_arm_wait_s: float = 120.0       # s, ceiling on waiting for that tick
    hw_init_timeout: float = 5.0        # max wait for low-level EtherCAT init [s]
    cores: tuple[int, int] = (2, 3)     # CPU affinity (low-level, control); Linux only
    rt_priority: tuple[int, int] = (80, 70)   # SCHED_FIFO; low-level outranks control

    low_warmup_s: float = 5.0           # s from the SYNCED ORIGIN to ec.enable

    allow_motion: bool = False          # gate on MOTION; False still enables + holds
    drive_knee: bool = False            # per-joint wire gate; False = drive holds
    drive_ankle: bool = False           # off while its torque loop rings near 42 Hz
    knee_torque_trip: float = 120.0      # de-energize + stop if |knee motor torque ff| exceeds [Nm]
    ankle_torque_trip: float = 120.0     # same, ankle [Nm]
    enable_grace_s: float = 2.0         # warn if not OPERATION_ENABLED this long after enable [s]
    shutdown_s: float = 1.0             # disable-ramp duration on teardown [s]
    max_stale_ticks: int = 5            # B3 watchdog: stale command -> safe zero

    # Scripted operator schedule [s from the synced origin]. The spawned control
    # process has no tty, so TerminalOperator cannot run there and this is the
    # only way to drive a run. bench_walk_at must fall after standing begins.
    bench_walk_at: float | None = 18.0
    bench_swing_at: tuple[float, ...] = ()      # () = no scripted swings
    bench_swing_duration: float = 1.0   # s, before the MJSWING clamp


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

    flex_ext: float                          # per-stop sign at angle_max_soft (+1)
    extend_ext: float                        # per-stop sign at angle_min_soft (-1)

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

    flex_ext: float                          # per-stop signs, see KneeParams
    extend_ext: float

    vel_sat_lowpass_cutoff: float            # Hz
    vel_sat_lowpass_order: int

    torque_ctrl_spring_vel_lowpass_cutoff: float   # Hz, spring-accel branch unported
    torque_ctrl_spring_vel_lowpass_order: int


# Encoders (spring_zero_positions.mat) -- linear scales, zeros, hall offsets.
# Motor counts-to-rad lives on the joint params. Used by Get Prosthesis State.
@dataclass(frozen=True, slots=True)
class EncoderParams:
    knee_abs_counts_to_rad: float            # empirical, sign-flipped; NOT 2pi/2^17
    ankle_abs_counts_to_rad: float           # 2*pi / 2^17 (17-bit abs)

    knee_abs_counts_offset: int              # knee0pos
    ankle_abs_counts_offset: int             # ankle0pos
    knee_motor_abs_counts_offset: int        # kneemotor0pos
    ankle_motor_abs_counts_offset: int       # anklemotor0pos

    hall_0_offset: int
    hall_1_offset: int


# IMU (two WitMotion HWT906, thigh + foot, over USB-serial). Ports can swap
# across replugs -- set them from imu_io.discover_imu_ports().
@dataclass(frozen=True, slots=True)
class IMUParams:
    thigh_port: str            # set from discover_imu_ports()
    foot_port: str
    baud: int                  # 921600 reliable on Linux
    stream_rate_hz: int        # sensor output rate
    lpf_cutoff_hz: int         # anti-alias; keep <= 0.4 * stream_rate
    acc_range_g: int           # +-g
    gyro_range_dps: int        # +-dps


# Encoder error correction (encoder_counts_to_rad_polys.mat). Normalize counts,
# eval the err poly, de-normalize. Polys are highest-power-first.
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


# Standing (baseline "Standing Mode Control"). Min-jerk ramp to a fixed pose
# over `duration`, then per-joint PD on the FILTERED velocity (angle stays raw).
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
    knee_k: float   # Nm/rad
    knee_b: float   # Nm/(rad/s)
    knee_off: float # rad
    ankle_k: float
    ankle_b: float
    ankle_off: float


# 3-state impedance stance: tau = -k*(angle - off) - b*angle_dot per joint,
# phases advancing on knee then ankle angle. Thresholds [rad].
@dataclass(frozen=True, slots=True)
class ImpedanceParams:
    stance_12_threshold: float
    stance_23_threshold: float
    stance1: ImpedanceStanceGains
    stance2: ImpedanceStanceGains
    stance3: ImpedanceStanceGains
    # Provenance only: the b values below are already floored. Do NOT re-apply.
    damping_lower_lim: float = 0.02


# Default instances (loaded once at import). Inject alternates in tests.
COMMON = CommonParams()
RUNTIME = RuntimeParams()

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
    angle_offset=0.0,                        # matches NPC params.m; old value 0.0444
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
    stream_rate_hz=1000,
    lpf_cutoff_hz=256,
    acc_range_g=16,
    gyro_range_dps=2000,
)

# From utility_functions/encoder_counts_to_rad_polys.mat (Feb 2024) -- the set
# paired with the Sep-2025 zero offsets. Polys, means and stds go together.
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
    kd=5.0, #5.0,
    knee_vel_lowpass_cutoff=25.0,
    ankle_vel_lowpass_cutoff=20.0,
    vel_lowpass_order=2,
)

# From optimized_gains_linear_v2.mat, a 20-vector demuxed in the baseline's
# "Split up parameters". Three b values were negative in the .mat and are
# stored here already floored -- what is in this file is what runs.
IMPEDANCE = ImpedanceParams(
    stance_12_threshold=0.18,   # rad, on knee_angle
    stance_23_threshold=0.05,   # rad, on ankle_angle
    stance1=ImpedanceStanceGains(
        knee_k=1.6952926155327914,
        knee_b=0.02,                      # clamped, .mat had -0.06311105716937081
        knee_off=0.04927967674031565,
        ankle_k=0.5813373023281654,
        ankle_b=0.02,                     # clamped, .mat had 0.013185618610104564
        ankle_off=-0.03604199280553995,
    ),
    stance2=ImpedanceStanceGains(
        knee_k=3.9978461205331,
        knee_b=0.06374872137254088,
        knee_off=0.20673632124501964,
        ankle_k=-1.1461063551070416,      # negative stiffness; baseline floors b only
        ankle_b=0.02,                     # clamped, .mat had -0.10424569658391881
        ankle_off=0.5561297042653888,
    ),
    stance3=ImpedanceStanceGains(
        knee_k=1.5746367312329719,
        knee_b=0.10278330936260308,
        knee_off=0.5390251724779882,
        ankle_k=2.5386451061873583,
        ankle_b=0.0509298768610955,
        ankle_off=-0.16748997556074638,
    ),
)


# Gait phase detection (WalkingControl). Hall GRF proxies are conditioned in
# ProprioceptiveEstimator; thresholds are in hall counts against a live
# unloaded zero, so the sign of the hall wiring does not matter.
@dataclass(frozen=True, slots=True)
class ContactParams:
    toe_threshold: float        # |toe_grf - baseline| above this = loaded
    heel_threshold: float       # |heel_grf - baseline| above this = loaded
    baseline_tau: float         # s, live-zero average, taken while UNLOADED only
    # OURS, not the baseline's: a refractory hold standing in for its leg-angle
    # guard until the IMU provides one. See docs/params.md before raising cadence.
    min_stance: float           # s, cannot leave stance before this
    min_swing: float            # s, cannot leave swing before this


CONTACT = ContactParams(
    toe_threshold=70.0,
    heel_threshold=80.0,
    baseline_tau=2.0,
    min_stance=0.30,
    min_swing=0.12,      # measured no benefit; kept as a knob
)


# Min-jerk swing (MinJerkSwing), from `minjerk swing Control/Min Jerk Swing` in
# nm_nn_swing. Two quintics per joint in swing-relative time, solved once at
# toe-off. Values from nm_nn.m 2.5 and params.m, NOT minjerk_params.m.
@dataclass(frozen=True, slots=True)
class MJSwingParams:
    swing_to_stance_ratio: float             # T = clamp(ratio * prev_stance, min, max)
    swing_duration_min: float                # s
    swing_duration_max: float                # s

    knee_flex_angle: float                   # rad, mid-swing flexion target
    knee_extend_angle: float                 # rad, end-of-swing target
    knee_flex_frac: float                    # knot time as a fraction of T
    knee_angle_min_soft: float               # rad, IC guard
    knee_angle_max_soft: float               # rad, IC guard

    ankle_flex_angle: float                  # rad
    ankle_extend_angle: float                # rad
    ankle_flex_frac: float                   # earlier than the knee's, for clearance
    ankle_angle_min_soft: float              # rad
    ankle_angle_max_soft: float              # rad

    winter_swing_duration: float             # s, the swing the knot accel was measured over
    winter_flex_knee_accel: float            # rad/s^2, negative: decelerating flexion

    # Discrete pid on the trajectory ERROR (not the measured velocity -- see
    # docs/params.md), forward Euler, no saturation, no anti-windup.
    knee_kp: float                           # Nm/rad
    knee_ki: float                           # Nm/(rad*s)
    knee_kd: float                           # Nm/(rad/s)
    ankle_kp: float
    ankle_ki: float
    ankle_kd: float
    deriv_filter_n: float                    # rad/s, PID derivative filter (N)

    knee_flex_vel: float = 0.0               # pinned by the formulation, not a knob
    ankle_flex_vel: float = 0.0


MJSWING = MJSwingParams(
    swing_to_stance_ratio=0.6,
    swing_duration_min=0.4,
    swing_duration_max=1.0,

    knee_flex_angle=60.0 * pi / 180.0,
    knee_extend_angle=10.0 * pi / 180.0,
    knee_flex_frac=0.5,
    knee_angle_min_soft=5.0 * pi / 180.0,
    knee_angle_max_soft=85.0 * pi / 180.0,

    ankle_flex_angle=5.0 * pi / 180.0,
    ankle_extend_angle=0.0,
    ankle_flex_frac=0.3,
    ankle_angle_min_soft=-20.0 * pi / 180.0,
    ankle_angle_max_soft=20.0 * pi / 180.0,

    winter_swing_duration=0.386,
    winter_flex_knee_accel=-84.265,

    knee_kp=30.0,
    knee_ki=145.0,
    knee_kd=6.5,
    ankle_kp=125.0,
    ankle_ki=125.0,
    ankle_kd=10.0,
    deriv_filter_n=100.0,
)


# NN swing (control/high_level/walking/nn_swing.py). Phase 1 plays the network's
# prediction, phase 2 extends onto the landing pose, phase 3 holds it. The
# network's own constants live with the weights, not here.
@dataclass(frozen=True, slots=True)
class NNSwingParams:
    trajectory_period: float    # s, played before the next trajectory replaces it
    knee_min: float             # rad, applied before the pid, not to the stored command
    knee_max: float             # rad
    descend_hold: float         # s a descending trajectory must play before handing over

    # Phase 2 (Xing eq 4.3-4.4): knee_vel = slope * hip_vel, slope ramping to
    # slope_target, driving the knee to hip_angle - hip_knee_offset.
    hip_knee_offset: float      # rad, the shank's forward lean at touchdown
    slope_target: float         # knee/hip velocity ratio the ramp heads for
    slope_step: float           # ramp increment per baseline tick, going down
    slope_step_up: float        # and going up, if phase 1 left it below target
    baseline_dt: float          # s, the tick slope_step was written for
    hip_vel_target: float       # rad/s, anti-stall floor; does NOT track cadence
    hip_angle_min: float        # rad, lower clamp so the target stays reachable
    decay_exponent: float       # >1, so the knee approaches the target asymptotically

    # Phase 3 (Xing eq 4.5 via the 4.6 blend), decay rates per baseline tick.
    gain_vel: float
    gain_accel: float

    knee_kp: float              # Nm/rad
    knee_ki: float              # Nm/(rad*s)
    knee_kd: float              # Nm/(rad/s)
    deriv_filter_n: float       # rad/s


NNSWING = NNSwingParams(
    trajectory_period=0.052,
    knee_min=2.0 * pi / 180.0,
    knee_max=85.0 * pi / 180.0,
    descend_hold=0.05,
    hip_knee_offset=20.0 * pi / 180.0,
    slope_target=-5.0,
    slope_step=0.05,
    slope_step_up=1.0,
    baseline_dt=0.001,
    hip_vel_target=3.0,
    hip_angle_min=26.0 * pi / 180.0,
    decay_exponent=1.2,
    gain_vel=0.01,
    gain_accel=0.04,
    knee_kp=30.0,
    knee_ki=100.0,
    knee_kd=15.0,
    deriv_filter_n=100.0,
)


# Mode FSM (ProsthesisControl). Mirrors the baseline's State Machine minus the
# terms needing the IMU; of those only `~contact` was safety-critical, and it
# is kept. See docs/params.md for the baseline guards.
@dataclass(frozen=True, slots=True)
class ModeParams:
    still_joint_velocity: float     # rad/s, encoder-only stand-in for |hip_vel| < 0.1
    enable_dwell: float             # s of continuous OPERATION_ENABLED before a policy runs
    operator_timeout: float         # s without an operator update -> HOLD


MODE = ModeParams(
    still_joint_velocity=0.15,
    enable_dwell=2.0,
    operator_timeout=0.25,
)


# dynamic_params.mat -> dynpars (9x1), for the high-level controller (not yet
# ported). MATLAB order preserved; semantic mapping unconfirmed -- label before use.
DYNPARS: tuple[float, ...] = (
    0.0742, 1.5952, 0.0064, 3.5057, 0.0896,
    2.5055, 0.0305, 0.0117, 0.0865,
)


# ESKF (estimation/eskf). Q diagonal entries are (sigma * dt)^2, per-body order
# [dtheta, gyro bias, pos (0), vel, accel bias]. From params_new.m.
@dataclass(frozen=True, slots=True)
class ESKFConfig:
    dt: float                    # nominal filter tick [s]; fallback when dt is not measurable
    dt_min_factor: float         # DEAD: caller clamps dt now; kept so the literal constructs
    dt_max_factor: float         # DEAD
    imu_stale_timeout: float     # s; an older sample means the USB stream stalled

    sigma_dtheta_thigh: float    # attitude random walk [rad/s]
    sigma_gyr_thigh: float       # gyro-bias walk
    sigma_dv_thigh: float        # velocity random walk [m/s^2]
    sigma_acc_thigh: float       # accel-bias walk
    sigma_dtheta_foot: float
    sigma_gyr_foot: float
    sigma_dv_foot: float
    sigma_acc_foot: float

    sigma_enc: float             # encoder angle noise [rad]
    sigma_rate: float            # joint-rate noise [rad/s]; raw derivative -> expect big
    sigma_model_pos: float       # kinematic-model floor, position closure [m]
    sigma_model_vel: float       # kinematic-model floor, velocity closure [m/s]
    sigma_zupt: float            # foot-velocity pseudo-measurement [m/s]

    p0_att: float                # initial error covariance, per-body VARIANCES [rad^2]
    p0_gyr_bias: float           # [(rad/s)^2]
    p0_pos: float                # [m^2]; 0 because the start pose defines the origin
    p0_vel: float                # [(m/s)^2]
    p0_acc_bias: float           # [(m/s^2)^2]

    init_duration: float         # s the comp filter runs before the ESKF starts
    comp_beta: float             # comp-filter pull rate [rad/s]
    dynamic_process_noise: bool  # inflate Q with the measured integration error
    correct_thigh_attitude: bool # observe thigh pitch from the velocity closure
    # Belt speed [m/s], positive forward; 0 overground. Enters with OPPOSITE
    # signs in the zupt and in the published x -- see docs/params.md.
    treadmill_speed: float


ESKF_CFG = ESKFConfig(
    dt=0.001,
    dt_min_factor=0.5,
    dt_max_factor=5.0,
    imu_stale_timeout=0.05,   # IMU streams ~200 Hz (5 ms); 50 ms = 10 missed
    # Allan-deviation characterization per HWT906, scaled to the 500 Hz loop.
    # sigma_dv is deliberately left INFLATED -- tune it on the replay harness.
    sigma_dtheta_thigh=0.0018, sigma_gyr_thigh=0.00054,
    sigma_dv_thigh=1.0, sigma_acc_thigh=0.011,
    sigma_dtheta_foot=0.0027, sigma_gyr_foot=0.00049,
    sigma_dv_foot=1.0, sigma_acc_foot=0.0045,
    sigma_enc=0.01,
    sigma_rate=0.1,
    sigma_model_pos=0.01,
    sigma_model_vel=0.01,
    sigma_zupt=0.01,
    p0_att=1e-4, p0_gyr_bias=1e-4, p0_pos=0.0, p0_vel=1e-3, p0_acc_bias=1e-3,
    init_duration=1.0,
    comp_beta=5.0,
    dynamic_process_noise=False,
    correct_thigh_attitude=True,
    treadmill_speed=0.0,
)


# Leg geometry for the ESKF kinematic chain. Decoded from the baseline ESKF
# transforms at identity attitude, lengths from params_new.m. Body frame is
# +x forward, +y left, +z up.
@dataclass(frozen=True, slots=True)
class LegGeometry:
    shank_length: float                          # ankle joint -> knee joint [m]
    knee_to_thigh_imu: tuple[float, float, float]  # knee -> thigh IMU origin, thigh frame [m]
    foot_imu_to_ankle: tuple[float, float, float]  # foot IMU origin -> ankle, foot frame [m]
    ankle_to_toe: tuple[float, float, float]     # ankle -> toe, foot frame [m]
    ankle_to_heel: tuple[float, float, float]    # ankle -> heel, foot frame [m]
    hip_to_knee: float                           # SUBJECT SPECIFIC [m]
    knee_angle_offset: float                     # baseline 'gamma' [rad]; UNUSED
    thigh_imu_pitch: float                       # IMU mount tilt [rad], TUNING


LEG = LegGeometry(
    shank_length=0.215,
    # Sensor-frame offset, NOT along the link: 5.5 cm off the sagittal plane,
    # 8.5 cm fore-aft, 10.4 cm above the knee. The baseline's alpha is already
    # baked in -- do not apply it again.
    knee_to_thigh_imu=(0.084567, 0.055, 0.103991),
    foot_imu_to_ankle=(0.0, 0.0, 0.0),       # CONFIRMED zero on our hardware
    ankle_to_toe=(0.156, 0.0, -0.096),
    ankle_to_heel=(-0.09, 0.0, -0.096),
    hip_to_knee=0.66,                        # 0.66 "John"; 0.71 "Russell"
    knee_angle_offset=6.0 * pi / 180.0,      # reference only; see docs/params.md
    # Our mount, not the baseline's gamma. Stays coupled to the same rot_y in
    # eskf.py and measurements.py -- one physical tilt, two effects.
    thigh_imu_pitch=7.0 * pi / 180.0,  #was -6
)


@dataclass(frozen=True, slots=True)
class Attachment:
    """One muscle's attachment at one joint, from the human muscle geometry.

    The moment arm falls from r_max at phi_max to r_min at phi_min. Note r is
    not d(dl)/d(phi): rho scales the length change but not the arm, which is
    deliberate in the source model. See docs/params.md.
    """
    r_max: float            # m, peak moment arm
    r_min: float            # m, arm at phi_min
    phi_max: float          # rad, joint angle of peak moment arm
    phi_min: float          # rad, joint angle at which the arm has fallen to r_min
    phi_ref: float          # rad, joint angle at which the MTU is at l_rest
    rho: float              # MTU length-change scaling (pennation etc.)
    sign: float             # +1 if the MTU lengthens with increasing phi
    constant_arm: bool = False   # hip-style: r = r_max, dl linear in phi

    @property
    def angle_scale(self) -> float:
        """`angle_scale_*` in nm_nn.m: acos(r_min/r_max)/(phi_min - phi_max)."""
        return acos(self.r_min / self.r_max) / (self.phi_min - self.phi_max)


@dataclass(frozen=True, slots=True)
class MuscleParams:
    name: str
    F_max: float                          # N/kg -- PER-KILOGRAM, scaled at the torque
    l_opt: float                          # m, optimal CE length
    l_slack: float                        # m, SE slack length
    v_max: float                          # l_opt/s -- multiply by l_opt for m/s
    eps_ref: float                        # SE reference strain; ankle plantarflexors differ
    prestim: float                        # baseline stimulation
    attachments: dict = None              # joint name -> Attachment; missing joint = no span


# Neuromuscular stance. Angles are the HUMAN anatomical convention the muscle
# tables are written in -- neuromuscular_stance converts at the boundary.
# Reflex-arc and joint-stop equations are in docs/params.md.
@dataclass(frozen=True, slots=True)
class NeuromuscularStanceParams:
    # shared muscle constants (nm_nn.m section 2)
    W: float           # CE force-length width
    C_RESID: float     # remaining force at +/- width; the CE gain is log(C_RESID)
    N: float           # eccentric force enhancement
    K_HILL: float      # force-velocity shape factor
    TAU: float         # excitation-contraction coupling, s
    DT: float          # muscle integration substep, s

    # neural transmission delays [s]: a pathway property, not a muscle one
    LONG: float        # distal / ankle group
    MEDIUM: float      # knee
    SHORT: float       # hip

    # stance reflex arcs (Muscle Stimulation/Stance Phase). VAS reads BOTH its
    # afferents at SHORT in the baseline, despite being a knee muscle.
    FGAIN_VAS: float
    LGAIN_VAS: float
    LCEOFF_VAS: float  # l_opt units
    FGAIN_GAS: float
    FGAIN_SOL: float
    DELAY_VAS: float   # s
    DELAY_GAS: float
    DELAY_SOL: float
    STIM_MIN: float
    STIM_MAX: float

    # Late-stance VAS suppression: the swing-initiation unlock. NEEDS HIP ANGLE,
    # so it is gated off until the ESKF provides one.
    LATE_STANCE_HIP: float     # rad
    VAS_SUPPRESS_EXP: float

    # Knee-only buckling stop + damper, on the REAL leg angle/velocity, both
    # per-kilogram. Summed inside Muscle Models, so they also pass through the
    # human->leg Jacobian -- matched to the baseline so the replay is comparable.
    KD_DAMPER: float
    KP_STOP: float
    KD_STOP: float
    STOP_ANGLE: float      # rad, knee angle below which the stop engages
    STOP_DELAY: float      # s after stance entry before the stop arms

    VAS: MuscleParams
    GAS: MuscleParams
    SOL: MuscleParams

    @property
    def EPS_BE(self) -> float:
        """BE reference strain: the baseline block is literally `w/2`."""
        return self.W / 2.0


# nm_nn.m section 2, with F_max and the four stance gains overridden by the
# tuned 7-vector in ParameterLibrary/nm_gains_new/nm_gains_1.mat.
NM_STANCE = NeuromuscularStanceParams(
    W=0.56,
    C_RESID=0.05,
    N=1.5,
    K_HILL=5.0,
    TAU=0.01,
    DT=0.0002,

    LONG=0.020,
    MEDIUM=0.010,
    SHORT=0.005,

    FGAIN_VAS=1.2,
    LGAIN_VAS=10.0,
    LCEOFF_VAS=0.625,
    FGAIN_GAS=1.67,     # nm_gains_1.mat; nm_nn.m literal is 1.6693
    FGAIN_SOL=1.25,
    DELAY_VAS=0.005,    # SHORT -- both VAS afferents, per the block diagram
    DELAY_GAS=0.020,    # LONG
    DELAY_SOL=0.020,    # LONG
    STIM_MIN=0.01,
    STIM_MAX=1.0,
    LATE_STANCE_HIP=0.0,        # `late stance threshold VAS` constant
    VAS_SUPPRESS_EXP=3.0,       # Fcn1 = u(1)^3

    KD_DAMPER=0.2,
    KP_STOP=7.0,
    KD_STOP=0.1,
    STOP_ANGLE=12.0*pi/180.0,   # Muscle Models/Constant1
    STOP_DELAY=0.200,           # Delay block, NumDelays=200 at 1 kHz

    VAS=MuscleParams(
        name="VAS",
        F_max=92.5,         # 7400/80
        l_opt=0.08,
        l_slack=0.23,
        v_max=12.0,
        eps_ref=0.04,
        prestim=0.12,
        attachments={
            "knee": Attachment(r_max=0.06, r_min=0.04,
                               phi_max=15.0*pi/180.0, phi_min=135.0*pi/180.0,
                               phi_ref=60.0*pi/180.0, rho=0.6,
                               sign=+1.0),      # knee extensor: lengthens with flexion
        },
    ),

    GAS=MuscleParams(
        name="GAS",
        F_max=16.875,       # 1350/80
        l_opt=0.05,
        l_slack=0.40,
        v_max=12.0,
        eps_ref=0.09,       # eref_ankle_pf
        prestim=0.01,
        attachments={
            "knee":  Attachment(r_max=0.05, r_min=0.02,
                                phi_max=40.0*pi/180.0, phi_min=135.0*pi/180.0,
                                phi_ref=15.0*pi/180.0, rho=0.7,
                                sign=-1.0),     # knee flexor: shortens with flexion
            "ankle": Attachment(r_max=0.03, r_min=0.02,
                                phi_max=-10.0*pi/180.0, phi_min=-90.0*pi/180.0,
                                phi_ref=10.0*pi/180.0, rho=0.7,
                                sign=+1.0),     # plantarflexor: lengthens with dorsiflexion
        },
    ),

    SOL=MuscleParams(
        name="SOL",
        F_max=37.5,         # nm_gains_1.mat; nm_nn.m literal is 2500/80 = 31.25
        l_opt=0.04,
        l_slack=0.26,
        v_max=6.0,
        eps_ref=0.09,       # eref_ankle_pf
        prestim=0.01,
        attachments={
            "ankle": Attachment(r_max=0.06, r_min=0.02,
                                phi_max=-10.0*pi/180.0, phi_min=-90.0*pi/180.0,
                                phi_ref=0.0, rho=0.5,
                                sign=+1.0),     # plantarflexor: lengthens with dorsiflexion
        },
    ),
)
