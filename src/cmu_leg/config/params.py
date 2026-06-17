"""
Source: baseline_controller/params.m and the .mat files it loads
(sys_id_knee_params, sys_id_ankle_params, ankle_stiffness_params,
spring_zero_positions, ankle_angle_offset, dynamic_params).

Angles in radians, torques in Nm, velocities in rad/s.

Parameters are grouped into frozen, slotted dataclasses (immutable at runtime,
no accidental gain mutation mid-loop) and instantiated once at module load into
the default instances KNEE / ANKLE / ENCODERS / STANDING / COMMON. Control blocks take a
params object in their constructor (defaulting to these module instances), so
the hot path still caches plain floats in __init__ while tests / per-subject
configs can inject alternates.

Joint / controller params (KneeParams / AnkleParams / EncoderParams /
StandingParams) are *required* fields
with no defaults: every value must be stated explicitly at instantiation so a
missing parameter is a loud TypeError, not a silent default. Only CommonParams
carries defaults.
"""
from dataclasses import dataclass
from math import pi


# ---------------------------------------------------------------------------
# Common / shared
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CommonParams:
    # Timing (params.m: sample_freq / sample_period / tstep)
    sample_rate: float = 1000.0              # Hz
    sample_time: float = 0.001               # s

    # Low Level Control startup mute: number of ticks after control_enabled
    # rises during which the pipeline runs (filters/derivatives settle) but the
    # drives are commanded to zero. Matches the "1 s Delay" on the drive-output
    # switches inside the Simulink "Low Level Control" subsystem
    # (DelayLength = sample_freq * 1 = 1000 samples at 1 kHz).
    startup_mute_ticks: int = 1000           # = round(1.0 * sample_rate)

    # Subject / body (params.m)
    gravity: float = 9.81                    # m/s^2
    swing_body_weight: float = 80.0          # kg (params.m: swing_body_weight)
    stance_body_weight: float = 75.0         # kg (params.m, sub 0 active)

    # Global limits / shared gains
    torque_rate_lim: float = 20.0            # Nm/s (params.m: torque_rate_lim)
    damping_gain: float = 0.15               # outer damping comp (both joints)

    # Default torque-conditioning lowpass (25 Hz / order 2). The conditioning
    # blocks currently construct Lowpass() with its own defaults; these fields
    # exist for when those blocks are migrated to injected params.
    lowpass_cutoff: float = 25.0             # Hz
    lowpass_order: int = 2


# ---------------------------------------------------------------------------
# Knee
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class KneeParams:
    # Motor / drivetrain
    torque_constant: float                   # Nm/A
    rated_current: float                     # A
    gear_ratio: float
    vel_max: float                           # rad/s, joint side
    motor_damping: float                     # sys_id_knee_params.mat
    motor_inertia: float
    motor_eff: float
    motor_abs_counts_to_rad: float           # 2*pi / 2^30

    # Spring / structural
    stiffness: float                         # Nm/rad

    # Control
    kp: float
    wc_ff: float                             # rad/s, Qff cutoff (params.m)
    wc_dob: float                            # rad/s, DOB cutoff (params.m)

    # Joint limits & soft stop
    angle_min: float
    angle_min_soft: float
    angle_max: float
    angle_max_soft: float
    torque_max: float                        # Nm
    c_jointstop: float                       # Nm/rad (soft-stop stiffness)
    w_max_jointstop: float                   # rad/s (soft-stop retract speed)

    # Flexion/extension stop sign multipliers. In the Simulink model these are
    # per-stop-block scalars (+1 at the upper limit, -1 at the lower), NOT a
    # single shared value. Wiring of these into KneeTorqueConditioning is still
    # pending (the conditioning block currently references a single unresolved
    # `ext`). See [[low-level-conditioning-ext-pending]].
    flex_ext: float                          # at angle_max_soft (+1)
    extend_ext: float                        # at angle_min_soft (-1)

    # Knee Velocity Saturation lowpass (Simulink "Generated Filter Block")
    vel_sat_lowpass_cutoff: float            # Hz
    vel_sat_lowpass_order: int

    @property
    def up_limit(self) -> float:
        """Alias kept for the conditioning block's `knee_up_limit`."""
        return self.angle_max_soft


# ---------------------------------------------------------------------------
# Ankle
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AnkleParams:
    # Motor / drivetrain
    torque_constant: float                   # Nm/A
    rated_current: float                     # A
    gear_ratio: float
    vel_max: float                           # rad/s, joint side
    motor_damping: float                     # sys_id_ankle_params.mat
    motor_inertia: float
    motor_eff: float                         # NOTE: efficiency comp disabled (arch.md)
    motor_abs_counts_to_rad: float           # 2*pi / 2^30

    # Spring / structural
    # `stiffness` is the linear scalar (params.m: 993.8613). `stiffness_poly`
    # holds the 2-coeff nonlinear fit from ankle_stiffness_params.mat, not yet
    # consumed by the controller.
    stiffness: float                         # Nm/rad (linear)
    stiffness_poly: tuple[float, float]      # Nm/rad poly coeffs
    angle_offset: float                      # rad

    # Control
    kp: float

    # Joint limits & soft stop
    angle_min: float
    angle_min_soft: float
    angle_max: float
    angle_max_soft: float
    torque_max: float                        # Nm
    c_jointstop: float                       # Nm/rad
    w_max_jointstop: float                   # rad/s

    # Flexion/extension stop sign multipliers (see KneeParams note).
    flex_ext: float
    extend_ext: float

    # Ankle Velocity Saturation lowpass ("Ankle Desired Saturation")
    vel_sat_lowpass_cutoff: float            # Hz
    vel_sat_lowpass_order: int

    # Ankle Torque Controller spring-velocity lowpass (20 Hz). Spring-accel
    # branch in Simulink is gain-0 and not ported.
    torque_ctrl_spring_vel_lowpass_cutoff: float   # Hz
    torque_ctrl_spring_vel_lowpass_order: int


# ---------------------------------------------------------------------------
# Encoders (spring_zero_positions.mat) — used by Get Prosthesis State.
# Joint/spring absolute-encoder scaling and all zero offsets. Motor-side
# counts-to-rad lives on the joint params (drivetrain scaling for drive output).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EncoderParams:
    knee_abs_counts_to_rad: float            # 2*pi / 2^17
    ankle_abs_counts_to_rad: float           # 2*pi / 2^17

    knee_abs_counts_offset: int              # knee0pos
    ankle_abs_counts_offset: int             # ankle0pos
    knee_motor_abs_counts_offset: int        # kneemotor0pos
    ankle_motor_abs_counts_offset: int       # anklemotor0pos

    hall_0_offset: int
    hall_1_offset: int


# ---------------------------------------------------------------------------
# High-level: Standing (baseline "Standing Mode Control" -> "Standing").
# Min-jerk quintic ramp from the latched start pose to a fixed standing pose
# over `duration`, then a per-joint PD hold:
#   tau = kp * (theta_traj - theta) - kd * theta_dot_filt
# Same gains for knee and ankle (baseline "Knee PD Control" Stifness/Damping).
#
# theta_dot_filt is the joint velocity passed through a per-joint Butterworth
# lowpass. Joint velocities leave Get Prosthesis State as raw finite differences
# (model annotation: "Kinematic Signals are unfiltered!!"); baseline "Standing
# Mode Control" lowpasses each velocity (NOT the angle) before the PD damping
# term. The stiffness term uses the raw, unfiltered angle.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StandingParams:
    knee_target: float                       # rad, target knee angle (knee_stand_angle)
    ankle_target: float                      # rad, target ankle angle (ankle_stand_angle)
    duration: float                          # s, min-jerk ramp duration (stand duration)
    kp: float                                # Nm/rad      (PD "Stifness")
    kd: float                                # Nm/(rad/s)  (PD "Damping")

    # PD damping-term velocity prefilter (3 dB point, Butterworth, SOS).
    knee_vel_lowpass_cutoff: float           # Hz, F3dB (baseline "Lowpass Filter2")
    ankle_vel_lowpass_cutoff: float          # Hz, F3dB (baseline "Lowpass Filter7")
    vel_lowpass_order: int                   # Butterworth order (both joints)


# ---------------------------------------------------------------------------
# Default instances (loaded once at import). Inject alternates in tests.
# ---------------------------------------------------------------------------
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
    knee_abs_counts_to_rad=2.0 * pi / (2 ** 17),
    ankle_abs_counts_to_rad=2.0 * pi / (2 ** 17),
    knee_abs_counts_offset=234_149_441,
    ankle_abs_counts_offset=-553,
    knee_motor_abs_counts_offset=-2_635,
    ankle_motor_abs_counts_offset=-951,
    hall_0_offset=1_738,
    hall_1_offset=2_261,
)

STANDING = StandingParams(
    knee_target=5.0 * pi / 180.0,            # baseline knee_stand_angle = 5 deg
    ankle_target=0.0,                        # baseline ankle_stand_angle = 0 deg
    duration=2.0,                            # baseline stand duration = 2 s
    kp=150.0,
    kd=5.0,
    knee_vel_lowpass_cutoff=25.0,            # baseline "Lowpass Filter2" F3dB
    ankle_vel_lowpass_cutoff=20.0,           # baseline "Lowpass Filter7" F3dB
    vel_lowpass_order=2,
)

# ---------------------------------------------------------------------------
# Dynamic parameters (dynamic_params.mat -> dynpars, 9x1). Used by the
# high-level (impedance / NM) controller, not yet ported. Order preserved from
# the MATLAB vector; semantic mapping not confirmed from the model — label
# before consuming downstream. Kept as a module-level immutable tuple until the
# high-level controller is ported and can own a dedicated dataclass.
# ---------------------------------------------------------------------------
DYNPARS: tuple[float, ...] = (
    0.0742, 1.5952, 0.0064, 3.5057, 0.0896,
    2.5055, 0.0305, 0.0117, 0.0865,
)
