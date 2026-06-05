"""
Source: baseline_controller/params.m and the .mat files it loads
(sys_id_knee_params, sys_id_ankle_params, ankle_stiffness_params,
spring_zero_positions, ankle_angle_offset, dynamic_params).

Angles in radians, torques in Nm, velocities in rad/s.
"""
from math import pi

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
SAMPLE_RATE: float = 1000.0              # Hz   (params.m: sample_freq)
SAMPLE_TIME: float = 0.001               # s    (params.m: sample_period / tstep)

# Low Level Control startup mute: number of ticks after control_enabled rises
# during which the pipeline runs (so filters and derivatives settle) but the
# drives are commanded to zero. Matches the "1 s Delay" on the drive-output
# switches inside the Simulink "Low Level Control" subsystem
# (DelayLength = sample_freq * 1 = 1000 samples at 1 kHz).
STARTUP_MUTE_TICKS: int = int(round(1.0 * SAMPLE_RATE))   # 1.0 s

# ---------------------------------------------------------------------------
# Subject / body
# ---------------------------------------------------------------------------
SWING_BODY_WEIGHT: float = 80.0          # kg   (params.m: swing_body_weight)
STANCE_BODY_WEIGHT: float = 75.0         # kg   (params.m, sub 0 active)
GRAVITY: float = 9.81                    # m/s^2

# ---------------------------------------------------------------------------
# Global limits
# ---------------------------------------------------------------------------
TORQUE_RATE_LIM: float = 20.0            # Nm/s (params.m: torque_rate_lim)

# ---------------------------------------------------------------------------
# Knee — motor / drivetrain
# ---------------------------------------------------------------------------
KNEE_TORQUE_CONSTANT: float = 0.065      # Nm/A  (params.m)
KNEE_RATED_CURRENT: float = 20.0         # A
KNEE_GEAR_RATIO: float = 50.0
KNEE_VEL_MAX: float = 5000.0 * 2.0 * pi / 60.0 / 50.0 * 0.95   # rad/s, joint side

# Identified motor model (sys_id_knee_params.mat)
KNEE_MOTOR_DAMPING: float = 9.7032
KNEE_MOTOR_INERTIA: float = 0.5340
KNEE_MOTOR_EFF: float = 0.9129

# ---------------------------------------------------------------------------
# Knee — joint limits & soft stop
# ---------------------------------------------------------------------------
KNEE_ANGLE_MIN: float = -2.0 * pi / 180.0
KNEE_ANGLE_MIN_SOFT: float = 5.0 * pi / 180.0
KNEE_ANGLE_MAX: float = 90.0 * pi / 180.0
KNEE_ANGLE_MAX_SOFT: float = 85.0 * pi / 180.0
KNEE_TORQUE_MAX: float = 100.0           # Nm

# Soft-stop (used by KneeTorqueConditioning)
#   c_jointstop      = knee_soft_stop_stiffness
#   w_max_jointstop  = knee_soft_stop_retract_speed
KNEE_C_JOINTSTOP: float = 160.0          # Nm/rad
KNEE_W_MAX_JOINTSTOP: float = 10.0       # rad/s

# Flexion/extension limit "ext" parameters in the Simulink low-level model are
# *per-block* sign multipliers — one for each of the 4 stop blocks
# (Knee Flex, Knee Ext, Ankle Flex, Ankle Ext). They are NOT a single shared
# scalar. The current Python conditioning collapses them into one `EXT` field,
# which is the source of the duplicated flexion/extension logic. Recommend
# splitting these out when wiring up the corrected stops.
#   sign convention in MATLAB model: +1 at the upper limit, -1 at the lower.
KNEE_FLEX_EXT: float = +1.0              # at knee_angle_max_soft
KNEE_EXTEND_EXT: float = -1.0            # at knee_angle_min_soft
KNEE_UP_LIMIT: float = KNEE_ANGLE_MAX_SOFT   # alias used by current code

# ---------------------------------------------------------------------------
# Knee — spring / structural
# ---------------------------------------------------------------------------
KNEE_STIFFNESS: float = 1666.0           # Nm/rad

# ---------------------------------------------------------------------------
# Knee — control
# ---------------------------------------------------------------------------
KNEE_KP: float = 0.09
DAMPING_GAIN: float = 0.15               # outer damping compensation (both joints)

# DOB / Qff cutoffs (params.m)
KNEE_WC_FF: float = 25.0 * 2.0 * pi      # rad/s
KNEE_WC_DOB: float = 10.0 * 2.0 * pi     # rad/s

# ---------------------------------------------------------------------------
# Ankle — motor / drivetrain
# ---------------------------------------------------------------------------
ANKLE_TORQUE_CONSTANT: float = 0.0529    # Nm/A
ANKLE_RATED_CURRENT: float = 15.7        # A
ANKLE_GEAR_RATIO: float = 100.0
ANKLE_VEL_MAX: float = 7000.0 * 2.0 * pi / 60.0 / 100.0 * 0.95  # rad/s

# Identified motor model (sys_id_ankle_params.mat)
ANKLE_MOTOR_DAMPING: float = 5.2961
ANKLE_MOTOR_INERTIA: float = 0.4683
ANKLE_MOTOR_EFF: float = 0.5583          # NOTE: efficiency comp disabled (arch.md)

# ---------------------------------------------------------------------------
# Ankle — joint limits & soft stop
# ---------------------------------------------------------------------------
ANKLE_ANGLE_MIN: float = -30.0 * pi / 180.0
ANKLE_ANGLE_MIN_SOFT: float = -20.0 * pi / 180.0
ANKLE_ANGLE_MAX: float = 30.0 * pi / 180.0
ANKLE_ANGLE_MAX_SOFT: float = 20.0 * pi / 180.0
ANKLE_TORQUE_MAX: float = 150.0          # Nm

ANKLE_C_JOINTSTOP: float = 80.0          # Nm/rad
ANKLE_W_MAX_JOINTSTOP: float = 5.0       # rad/s

ANKLE_FLEX_EXT: float = +1.0
ANKLE_EXTEND_EXT: float = -1.0

# ---------------------------------------------------------------------------
# Ankle — spring / structural
# ---------------------------------------------------------------------------
# ankle_stiffness in params.m is the scalar 993.8613 (linear);
# ankle_stiffness_params.mat holds the 2-coeff polynomial
# [1.0804e3, -6.2165e3] for the nonlinear fit (currently NOT loaded by params.m).
ANKLE_STIFFNESS: float = 993.8613                       # Nm/rad (linear)
ANKLE_STIFFNESS_POLY: tuple[float, float] = (1080.4, -6216.5)  # Nm/rad poly coeffs
ANKLE_ANGLE_OFFSET: float = 0.0444                      # rad

# ---------------------------------------------------------------------------
# Ankle — control
# ---------------------------------------------------------------------------
ANKLE_KP: float = 0.08

# ---------------------------------------------------------------------------
# Encoders (spring_zero_positions.mat)
# ---------------------------------------------------------------------------
KNEE_ABS_COUNTS_TO_RAD: float = 2.0 * pi / (2 ** 17)
ANKLE_ABS_COUNTS_TO_RAD: float = 2.0 * pi / (2 ** 17)
KNEE_MOTOR_ABS_COUNTS_TO_RAD: float = 2.0 * pi / (2 ** 30)
ANKLE_MOTOR_ABS_COUNTS_TO_RAD: float = 2.0 * pi / (2 ** 30)

KNEE_ABS_COUNTS_OFFSET: int = 234_149_441    # knee0pos
ANKLE_ABS_COUNTS_OFFSET: int = -553          # ankle0pos
KNEE_MOTOR_ABS_COUNTS_OFFSET: int = -2_635   # kneemotor0pos
ANKLE_MOTOR_ABS_COUNTS_OFFSET: int = -951    # anklemotor0pos

HALL_0_OFFSET: int = 1_738
HALL_1_OFFSET: int = 2_261

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
LOWPASS_CUTOFF: float = 25.0             # Hz   used by torque conditioning
LOWPASS_ORDER: int = 2

# Knee Velocity Saturation block uses a separate, faster lowpass on both the
# saturated velocity and the saturated torque-FF outputs (extracted from the
# Simulink "Generated Filter Block" coefficients).
KNEE_VEL_SAT_LOWPASS_CUTOFF: float = 75.0   # Hz
KNEE_VEL_SAT_LOWPASS_ORDER: int = 2

# Ankle Torque Controller filters spring velocity at 20 Hz before adding it
# into the velocity-command path. (Spring-accel branch in Simulink is gain-0
# and therefore not ported.)
ANKLE_TORQUE_CTRL_SPRING_VEL_LOWPASS_CUTOFF: float = 20.0   # Hz
ANKLE_TORQUE_CTRL_SPRING_VEL_LOWPASS_ORDER: int = 2

# Ankle Velocity Saturation (Simulink: "Ankle Desired Saturation") uses 50 Hz
# lowpass on both saturated velocity and saturated torque-FF outputs.
ANKLE_VEL_SAT_LOWPASS_CUTOFF: float = 50.0   # Hz
ANKLE_VEL_SAT_LOWPASS_ORDER: int = 2

# ---------------------------------------------------------------------------
# Dynamic parameters (dynamic_params.mat -> dynpars, 9x1)
# Used by the high-level (impedance / NM) controller. Order preserved from the
# MATLAB vector; semantic mapping not yet confirmed from the model — flag and
# label before consuming downstream.
# ---------------------------------------------------------------------------
DYNPARS: tuple[float, ...] = (
    0.0742, 1.5952, 0.0064, 3.5057, 0.0896,
    2.5055, 0.0305, 0.0117, 0.0865,
)
