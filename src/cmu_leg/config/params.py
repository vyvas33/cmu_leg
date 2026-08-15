"""Controller parameters, ported from baseline_controller/params.m and its .mat
files (sys_id_*, ankle_stiffness_params, spring_zero_positions, encoder_counts_
to_rad_polys, ankle_angle_offset, dynamic_params).
"""
from dataclasses import dataclass
from math import pi, acos


# Common / shared
@dataclass(frozen=True, slots=True)
class CommonParams:
    sample_rate: float = 1000.0              # Hz, low-level loop rate
    sample_time: float = 0.001               # s

    # Control-process loop rate (merged ESKF + high-level; can't hold 1 kHz on the
    # Jetson). Fixed-coefficient filters ticking in that process (standing velocity
    # lowpasses) MUST be designed at this rate, not sample_rate, or their effective
    # cutoff is wrong. launch.py derives DT_CONTROL from it.
    control_sample_rate: float = 500.0       # Hz, control-process loop rate

    # Measured-dt clamp for every differentiator/integrator. dt_max stops a
    # scheduler-stall tick (~21 ms observed) integrating as one step; dt_min bounds
    # 1/dt quantization-noise gain on the velocity derivatives.
    dt_min: float = 0.0005                   # s
    dt_max: float = 0.005                    # s

    # post-enable ticks the pipeline runs muted (filters settle); baseline LLC
    # "1 s Delay" = 1000 @ 1 kHz
    startup_mute_ticks: int = 1000

    gravity: float = 9.81                    # m/s^2
    swing_body_weight: float = 80.0          # kg
    stance_body_weight: float = 75.0         # kg

    torque_rate_lim: float = 20.0            # Nm/s
    damping_gain: float = 0.15               # outer damping comp (both joints)

    # default torque-conditioning lowpass; unused until those blocks take params
    lowpass_cutoff: float = 25.0             # Hz
    lowpass_order: int = 2


# Multiprocess runtime (runtime/launch.py + low_level.py + control.py). Loop
# periods derive from the rates in CommonParams (DT = 1/rate); these are the
# orchestration + safety knobs that used to live scattered across those files.
@dataclass(frozen=True, slots=True)
class RuntimeParams:
    # Experiment duration, counted from the FIRST ACTUATING TICK -- the moment
    # the mode FSM first reaches STANDING or WALKING -- not from the synced
    # origin. Warmup no longer eats into the run, and with TerminalOperator the
    # start is not predictable at all, so a fixed window from t0 would give an
    # arbitrary amount of usable data depending on how long the operator took.
    tmax: float = 20.0                  # experiment duration [s]
    # Ceiling on the wait for that first actuating tick. If the operator never
    # asks (or the drives never enable), the run ends here instead of spinning.
    max_arm_wait_s: float = 120.0
    hw_init_timeout: float = 5.0        # max wait for low-level EtherCAT init [s]
    cores: tuple[int, int] = (2, 3)     # CPU affinity (low-level, control); Linux only
    # SCHED_FIFO priorities for the same two, applied by each child itself.
    # Low-level outranks control: it holds the 1 kHz EtherCAT cycle, and a stall
    # past the drives' sync-manager watchdog (measured between 2.5 and 2.8 ms)
    # quick-stops them. Both sit below 99 so kernel threads landing on the core
    # can still pre-empt. Needs root or CAP_SYS_NICE; without it the child warns
    # and continues at default priority.
    rt_priority: tuple[int, int] = (80, 70)

    # When ec.enable is commanded, measured from the SYNCED ORIGIN -- i.e. from
    # the moment control finishes its IMU bring-up and stamps start_time, NOT
    # from launch. Wall-clock is therefore (EtherCAT init + IMU init + this).
    # It no longer has to clear anything on the control side: the mode FSM waits
    # for state.control_enabled (the drives actually reaching OPERATION_ENABLED),
    # so an enable walk that runs long delays standing instead of being ramped
    # into. Logistic time for the operator to reach the e-stop is this plus
    # control_warmup_s below -- both count from the same origin.
    low_warmup_s: float = 10.0
    # (control_warmup_s removed: standing is now the resting operator request,
    # matching the baseline's standing_mode_toggle Constant of 1, so there is
    # nothing left to schedule. The logistic wait is low_warmup_s above -- the
    # drives energize, control_enabled arrives, standing begins.)

    # Bring-up gate on MOTION, not on commanding. False still energizes the
    # drives, still walks CiA-402 to OPERATION_ENABLED, still runs the whole
    # mode FSM and low-level pipeline -- it just substitutes a zero
    # DriveCommand, which in CSV is target-velocity-zero, i.e. the drives hold.
    # So the leg is stiff, not limp, and nothing MOVES. (Was `command_motors`,
    # which read as "do not command the motors" when we very much still do.)
    allow_motion: bool = False
    # Per-joint drive gate, applied at the wire in low_level.run(). False sends a
    # zero DriveCommand for that joint -- which in CSV is target velocity zero,
    # so the drive holds it rather than leaving it free. Everything upstream
    # still runs and still logs, so the joint's commanded torque stays visible.
    # Ankle off while its torque loop limit-cycles near 42 Hz.
    drive_knee: bool = False
    drive_ankle: bool = False
    knee_torque_trip: float = 120.0      # de-energize + stop if |knee motor torque ff| exceeds [Nm]
    ankle_torque_trip: float = 120.0     # same, ankle [Nm]
    enable_grace_s: float = 2.0         # warn if not OPERATION_ENABLED this long after enable [s]
    shutdown_s: float = 1.0             # disable-ramp duration on teardown [s]
    max_stale_ticks: int = 5            # B3 watchdog: stale command -> safe zero

    # Scripted operator schedule, in seconds from the synced origin. The control
    # process is spawned, and multiprocessing replaces a child's stdin with
    # devnull, so TerminalOperator can never come up there and the schedule is
    # the only way to drive a run.
    #
    # bench_walk_at must fall AFTER standing has actually begun, which is
    # low_warmup_s + however long the CiA-402 enable walk takes + enable_dwell.
    # Requesting walking earlier drops standing_toggle while the mode FSM is
    # still in INITIALIZING, whose guard needs it true, and the run then sits
    # there for good. Standing typically lands near 12.5 s; 18.0 leaves margin
    # for a slow enable walk.
    #
    # bench_swing_at is the scripted stand-in for the space key, for a leg in a
    # harness where the foot never loads and no toe-off is ever produced.
    # Entering walking already fires one swing of its own, so the first entry
    # here is spaced clear of it. tmax runs from the first actuating tick, so
    # the last entry has to leave room for its own swing to finish.
    bench_walk_at: float | None = 18.0
    bench_swing_at: tuple[float, ...] = (20.0, 22.0, 24.0, 26.0, 28.0, 30.0)
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
    # Damping floor the baseline applies to all six b terms on the way in
    # (six Saturate blocks, [lim, inf], inside "Split up parameters"). The
    # b values below are already clamped, so this is provenance for anyone
    # reloading a different .mat -- ImpedanceStance must not re-apply it.
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
    kd=5.0, #5.0,
    knee_vel_lowpass_cutoff=25.0,
    ankle_vel_lowpass_cutoff=20.0,
    vel_lowpass_order=2,
)

# 3-state finite-state impedance stance (ImpedanceStance). Each phase applies
#   tau = -k*(angle - off) - b*angle_dot   for knee and ankle.
# Phase advance: Stance1 -> Stance2 when knee_angle > stance_12_threshold,
#                Stance2 -> Stance3 when ankle_angle > stance_23_threshold.
#
# Source: impedance_control_basic/ParameterLibrary/impedance_control/
#   optimized_gains_linear_v2.mat -- a 20-vector demuxed in the baseline's
#   "Split up parameters" block. Index order verified against that demux
#   (2026-08-10): 1-12 k/b interleaved per stance, 13-18 offsets, 19-20
#   thresholds. v2 differs from optimized_gains_linear.mat only in
#   stance_12_threshold (0.18 vs 0.3649); every other element is identical.
#
# Units: k [Nm/rad], b [Nm/(rad/s)], off and thresholds [rad].
#
# The three b values marked "clamped" were NEGATIVE in the .mat (optimizer
# output). The baseline floors all six at damping_lower_lim on load, so the
# joints never see them; the clamped values are stored here for the same
# reason -- what is in this file is what runs.
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
        ankle_k=-1.1461063551070416,      # negative stiffness; baseline floors b only, not k
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
# ProprioceptiveEstimator; the threshold and the FSM live in the controller,
# because `stance` never goes on the bus (see types/state.py).
#
# The baseline (`Get Prosthesis State/ini SM` + `SM`) auto-tares the halls by
# averaging them WHILE UNLOADED and comparing against that live zero, which
# absorbs hall drift between sessions. Thresholds are in hall-count units and
# abs() is applied, so the sign of the hall wiring does not matter.
@dataclass(frozen=True, slots=True)
class ContactParams:
    toe_threshold: float        # |toe_grf - baseline| above this = loaded
    heel_threshold: float       # |heel_grf - baseline| above this = loaded
    # Live-zero tracking: first-order average of the GRF while UNLOADED only.
    # tau is in seconds; a long tau means the zero follows drift but not steps.
    baseline_tau: float
    # OURS, not the baseline's. The baseline de-chatters its FSM with a
    # leg-angle guard (enter stance only if leg_angle < 85 deg), which needs
    # the IMU. Until that exists, a refractory hold: once a phase is entered it
    # cannot be left for this long. Chosen over a symmetric debounce because a
    # debounce delays the heel-strike edge too, clipping stance duty; a hold
    # only suppresses spurious re-triggers. On the Oct-2025 trial min_stance
    # 0.30 s takes heel strikes from 16 (chatter) to 7 (correct) and agreement
    # with the baseline stance flag from 0.940 to 0.965.
    #
    # CAUTION: 0.30 s is ~35% of the stance duration in that trial (0.85-1.47 s).
    # At a markedly faster cadence it would start truncating real stances --
    # revisit if walking speed goes up, and delete it once the leg-angle guard
    # is available.
    min_stance: float           # s, cannot leave stance before this
    min_swing: float            # s, cannot leave swing before this


CONTACT = ContactParams(
    toe_threshold=70.0,
    heel_threshold=80.0,
    baseline_tau=2.0,
    min_stance=0.30,
    min_swing=0.0,      # measured no benefit; kept as a knob
)


# Min-jerk swing (MinJerkSwing). From `minjerk swing Control/Min Jerk Swing` in
# nm_nn_swing; values from nm_nn.m section 2.5 and params.m. Read parameters from
# those, not from minjerk_params.m in the same folder, which is an unused older
# script holding different values.
#
# Per joint the trajectory is two quintics in swing-relative time, solved once at
# toe-off from the measured toe-off state:
#   segment 1  [0, flex_frac*T]  (theta_0, dtheta_0, ddtheta_0) -> (flex, 0, a_knot)
#   segment 2  [flex_frac*T, T]  (flex, 0, a_knot)              -> (extend, 0, 0)
# Past T a clamp holds it at (extend, 0, 0), which the segment-2 end constraints
# already pin.
#
# The knee pins its knot acceleration to a cadence-scaled Winter value,
# a_knot = winter_flex_knee_accel / (T / winter_swing_duration)**2, which leaves
# both knee segments fully determined. The ankle constrains only continuity of
# the knot acceleration, leaving one free degree of freedom that the jerk cost
# resolves.
@dataclass(frozen=True, slots=True)
class MJSwingParams:
    # Swing duration adapts to cadence off the PREVIOUS stance duration:
    #   T = clamp(swing_to_stance_ratio * prev_stance_duration, min, max)
    swing_to_stance_ratio: float
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

    # Knee knot acceleration, from Winter's Appendix A4 relative joint angular
    # kinematics: the mean of the two samples straddling peak knee flexion, over a
    # swing of winter_swing_duration. Time-scaled to T by the 1/t^2 law.
    winter_swing_duration: float             # s
    winter_flex_knee_accel: float            # rad/s^2 (negative: decelerating flexion)

    # Trajectory tracking. Discrete pid on the error e = theta_traj - theta,
    # forward Euler on both integrator and derivative filter, no output saturation
    # and no anti-windup; the low-level torque conditioning bounds the command.
    # Gains are absolute Nm/rad, since the baseline's stance/swing mixing scales
    # only the stance branch by body weight.
    #
    # The d term acts on the error, not on the measured velocity as in
    # StandingController. Swing theta_dot_traj peaks near 6.5 rad/s, where damping
    # the measurement would brake the commanded flexion instead of correcting lag.
    knee_kp: float                           # Nm/rad
    knee_ki: float                           # Nm/(rad*s)
    knee_kd: float                           # Nm/(rad/s)
    ankle_kp: float
    ankle_ki: float
    ankle_kd: float
    deriv_filter_n: float                    # rad/s, PID derivative filter (N)

    # Pinned by the formulation, not a tuning knob: the knot is the flexion peak,
    # so velocity there is zero by definition on both sides.
    knee_flex_vel: float = 0.0
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


# Mode FSM (ProsthesisControl). Mirrors the baseline's
# `Get Prosthesis State/State Machine`, minus the terms that need the IMU.
#
# Baseline guards, for reference:
#   StandingMode -> Swing   [~contact && ~toggle && hip_velocity > 0
#                            && leg_angle > 90 deg]
#   Stance -> StandingMode  [toggle && |hip_velocity| < 0.1
#                            && |leg_angle - pi/2| < 10 deg]
#
# Of those, only `~contact` is SAFETY-critical -- it is what guarantees walking
# is entered unloaded, so the knee stop can never fire against a straight
# standing knee. The hip_velocity / leg_angle terms are INTENT detection (is
# the wearer actually trying to walk). Without the ESKF we keep the guarantee
# and lose the intent, which means the operator has to be deliberate about when
# they flip the toggle, but nothing unsafe can happen.
@dataclass(frozen=True, slots=True)
class ModeParams:
    # "nearly still", the encoder-only stand-in for |hip_velocity| < 0.1.
    # Joint velocities are all we have without the IMU.
    still_joint_velocity: float     # rad/s, both joints must be under this
    # How long the drives must hold OPERATION_ENABLED before a policy may run.
    # Through this window the mode is still INITIALIZING, so mode.actuate is
    # False and the drives receive real zero commands -- their internal loops
    # settle on a zero-velocity target, and a bouncing enable shows up here
    # instead of underneath a trajectory. Standing's min-jerk clock only starts
    # afterwards, so the whole ramp is actually delivered.
    enable_dwell: float             # s
    # An operator source that stops updating for this long is treated as gone
    # and the FSM drops to HOLD. Mirrors RuntimeParams.max_stale_ticks
    # on the command buffer.
    operator_timeout: float         # s


MODE = ModeParams(
    still_joint_velocity=0.15,
    enable_dwell=2.0,
    operator_timeout=0.25,
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
    # DEAD: the caller now passes a dt already clamped to COMMON.dt_min/dt_max, so
    # the ESKF no longer clamps. Kept only so the literal below still constructs;
    # delete both once nothing references them.
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
    # correct thigh ATTITUDE from the velocity loop closure (the +R_th@hat(u)
    # Jacobian block the draft dropped). Off = baseline behavior (thigh pitch
    # rides gyro integration only, drifts); on = pitch is observed during swing.
    correct_thigh_attitude: bool


ESKF_CFG = ESKFConfig(
    dt=0.001,
    dt_min_factor=0.5,
    dt_max_factor=5.0,
    imu_stale_timeout=0.05,   # IMU streams ~200 Hz (5 ms); 50 ms = 10 missed
    # White terms (dtheta, dv) and bias walks (gyr, acc) from Allan-deviation
    # characterization of each HWT906 (10 min stationary, worst axis), scaled
    # sigma = C * sqrt(f_loop) with f_loop = 500 Hz (ESKF prediction rate).
    # dv is left INFLATED at the old 1.0 -- the ~0.021 Allan floor is the
    # stationary sensor VRW only; a swinging leg needs velocity distrusted well
    # above that. Tune dv on the replay harness, not from Allan.
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
    thigh_imu_pitch=7.0 * pi / 180.0,  #was -6
)

@dataclass(frozen=True, slots=True)
class Attachment:
    """One muscle's attachment at one joint, from the human muscle geometry.

    The moment arm falls from r_max at phi_max to r_min at phi_min:
        r(phi)  = r_max * cos(a * (phi - phi_max))
        dl(phi) = (rho*r_max/a) * [sin(a*(phi-phi_max)) - sin(a*(phi_ref-phi_max))]
    with a = angle_scale. Note r is not d(dl)/d(phi): rho scales the length
    change but not the moment arm. That asymmetry is in the source model
    deliberately -- keep it. The sign of `a` does not matter (cos and
    sin(ax)/a are both even in a).
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
    # PER-KILOGRAM (N/kg). nm_nn.m writes these as e.g. 7400/80; the baseline
    # multiplies stance torque by stance_body_weight in NM stance Control
    # (Gain/Gain1), so keep the normalisation here and scale at the torque.
    F_max: float                          # N/kg
    l_opt: float                          # m, optimal CE length
    l_slack: float                        # m, SE slack length
    v_max: float                          # l_opt/s -- multiply by l_opt for m/s
    eps_ref: float                        # SE reference strain; ankle plantarflexors differ
    prestim: float                        # baseline stimulation, [] 
    # joint name -> Attachment. Keys must match the joints the controller
    # passes in ("knee", "ankle"); a missing joint means the muscle does not
    # span it, so its moment arm there is zero.
    attachments: dict = None


@dataclass(frozen=True, slots=True)
class NeuromuscularStanceParams:
    # shared muscle constants (nm_nn.m section 2)
    W: float           # CE force-length width
    C_RESID: float     # remaining force at +/- width; the CE gain is log(C_RESID)
    N: float           # eccentric force enhancement
    K_HILL: float      # force-velocity shape factor
    TAU: float         # excitation-contraction coupling, s
    DT: float          # muscle integration substep, s

    # neural transmission delays, s. A pathway property, not a muscle one --
    # the reflex arcs below pick which one they read at.
    LONG: float        # distal / ankle group
    MEDIUM: float      # knee
    SHORT: float       # hip

    # stance reflex arcs (Muscle Stimulation/Stance Phase). Verified against
    # the block diagram 2026-08-10:
    #   stim_vas = prestim_vas + FGAIN_VAS*(F_vas(t-d)/F_max_vas)
    #                          + clamp(LGAIN_VAS*(l_ce_vas(t-d)/l_opt_vas - LCEOFF_VAS), -1, 0)
    #   stim_gas = prestim_gas + FGAIN_GAS*(F_gas(t-d)/F_max_gas)
    #   stim_sol = prestim_sol + FGAIN_SOL*(F_sol(t-d)/F_max_sol)
    # then all clamped to [STIM_MIN, STIM_MAX].
    # VAS reads BOTH its afferents at SHORT (5 ms) in the baseline -- not
    # MEDIUM, despite being a knee muscle. GAS and SOL read at LONG (20 ms).
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

    # Late-stance VAS suppression (Stimulation Mixing Prosthesis). Once the
    # thigh passes behind LATE_STANCE_HIP and GAS force falls off its peak for
    # this stance, VAS stim is scaled by (F_gas/max F_gas)^VAS_SUPPRESS_EXP,
    # clamped to [0,1]. This is the swing-initiation unlock: it lets the knee
    # break into flexion. NEEDS HIP ANGLE, so it is gated off until the ESKF
    # provides one -- see NeuromuscularStance.
    LATE_STANCE_HIP: float     # rad
    VAS_SUPPRESS_EXP: float

    # Knee-only joint stop + damper, on the REAL leg angle/velocity. Verified
    # against Muscle Models 2026-08-10:
    #   damper = -KD_DAMPER*knee_vel                       if knee_vel < 0
    #   stop   = (KP_STOP*(STOP_ANGLE - knee)
    #             - KD_STOP*knee_vel)                      if knee < STOP_ANGLE
    #                                                      and STOP_DELAY elapsed
    # The stop is a buckling guard: it pushes the knee back toward flexion when
    # it gets straighter than STOP_ANGLE. Both are per-kilogram like the muscle
    # forces. The baseline sums them INSIDE Muscle Models, so they also pass
    # through the human->leg Jacobian -- arguably wrong for a hardware stop, but
    # we match it so the replay is comparable.
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


# Source: prosthesis_control_NPC/nm_nn.m (section 2) + the tuned 7-vector in
# ParameterLibrary/nm_gains_new/nm_gains_1.mat, which overrides the three F_max
# values: Gains = [fmax_vas, fmax_gas, fmax_sol, st_fgain_vas, st_lgain_vas,
# st_fgain_gas, st_fgain_sol] = [92.5, 16.875, 37.5, 1.2, 10, 1.67, 1.25].
#
# Angles are the HUMAN anatomical convention the muscle tables are written in,
# NOT hardware angles -- neuromuscular_stance converts at the boundary. The
# leg->human kinematic remap is still open (see the module docstring).
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

