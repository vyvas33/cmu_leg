"""ESKF measurement models: each returns (r, V) for one sequential update
(the velocity closure additionally returns a 3x3 thigh-attitude H block).

Most H are a 3x3 identity selecting one error-state block, so it is not built or
returned -- eskf._kalman_update keys off the block start index (TH_DP/TH_DV/FO_DV).
The velocity closure is the exception: its residual also depends on thigh
attitude, so it returns that extra Jacobian block for _kalman_update's `extra`.

DRAFT -- verify signs, the error-state layout, and the V constructions before
trusting. Runtime path only: numpy, no sympy. The closed forms here are
transcribed from the verified symbolic derivation in kinematics.py, which stays
the source of truth.

Conventions (all established in kinematics.py):
  - Foot body frame: +x forward, +z up, +y left. Joint axis is +y.
  - Chain joint angles relate to encoder angles by an affine map:
        theta = S_JOINT * <encoder angle> + THETA0_JOINT
    with S = -1 for both joints (both CONFIRMED on hardware), THETA0 = 0.
  - q is body->nav; R = q_to_R(q). Attitude error is LOCAL/right:
        R_true = R_nom @ (I + hat(dtheta)),  matching _predict's q ⊗ Δq_body.
"""

import numpy as np

from cmu_leg.estimation.eskf.helpers import q_to_R, hat
from cmu_leg.config.params import LEG

def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# ---------------------------------------------------------------------------
# Geometry, from params.LEG (decoded from the baseline ESKF transforms).
# P_SH_TH is a full 3-vector, NOT an offset along the link: the thigh IMU sits
# off the sagittal plane. Its y term is invariant under rot_y, so it only
# shifts p_fo_th[1] by a constant and leaves the Jacobians' y rows zero.
# P_FO_ANKLE is zero -- the foot IMU sits on the ankle joint (confirmed).
# ---------------------------------------------------------------------------
L_SHANK = LEG.shank_length                        # ankle joint -> knee joint [m]
P_FO_ANKLE = np.array(LEG.foot_imu_to_ankle)      # foot IMU origin -> ankle, foot frame [m]

# knee -> thigh IMU origin. LEG.knee_to_thigh_imu is in the thigh SENSOR frame,
# so it takes the same rot_y(-pitch) that eskf.py's R_BODY_SENSOR_TH applies to
# the thigh measurements -- one physical mount tilt, both effects. Changing the
# pitch in params.LEG therefore moves the knee AND the chain frame together;
# that coupling is the point, do not decouple them.
P_SH_TH = _rot_y(-LEG.thigh_imu_pitch) @ np.array(LEG.knee_to_thigh_imu)

# encoder -> chain angle map (see kinematics.py). Both signs CONFIRMED.
S_ANKLE, THETA0_ANKLE = -1.0, 0.0
S_KNEE,  THETA0_KNEE = -1.0, 0.0

# ---------------------------------------------------------------------------
# Error-state layout (30) -- thigh block then foot block, 15 each.
# Sub-order mirrors the nominal fields: [dtheta, dw_b, dp, dv, da_b].
# Every H keys off these indices; A/P must use the same order.
# ---------------------------------------------------------------------------
def _blk(base):
    return dict(dtheta=slice(base + 0, base + 3),
                dw_b=slice(base + 3, base + 6),
                dp=slice(base + 6, base + 9),
                dv=slice(base + 9, base + 12),
                da_b=slice(base + 12, base + 15))

TH = _blk(0)
FO = _blk(15)
N = 30


def _chain_angles(enc):
    """Encoder readings -> (theta_ankle, theta_knee) in chain convention."""
    a = S_ANKLE * enc.ankle_angle + THETA0_ANKLE
    k = S_KNEE * enc.knee_angle + THETA0_KNEE
    return a, k


# ---------------------------------------------------------------------------
# Forward kinematics + Jacobians (foot coordinates), transcribed from sympy.
# ---------------------------------------------------------------------------
def fk_p_fo_th(a, k):
    """Thigh body origin in foot body frame. a,k are CHAIN angles."""
    return P_FO_ANKLE + _rot_y(a) @ np.array([0.0, 0.0, L_SHANK]) \
        + _rot_y(a + k) @ P_SH_TH


def pos_jacobian(a, k):
    """d p_fo_th / d[theta_ankle, theta_knee], 3x2, chain angles. (== J_pos)"""
    tx, _, tz = P_SH_TH
    ca, sa = np.cos(a), np.sin(a)
    cak, sak = np.cos(a + k), np.sin(a + k)
    d_ak_x = -tx * sak + tz * cak
    d_ak_z = -tx * cak - tz * sak
    return np.array([
        [L_SHANK * ca + d_ak_x,  d_ak_x],
        [0.0,                    0.0],
        [-L_SHANK * sa + d_ak_z, d_ak_z],
    ])


def body_jacobian(a, k):
    """Body Jacobian J_b (6x2), V_b_ft = J_b @ theta_dot in THIGH body frame.
    Rows [v; omega], MLS ordering. Depends on k only (offset-independent)."""
    tx, _, tz = P_SH_TH
    return np.array([
        [L_SHANK * np.cos(k) + tz, tz],
        [0.0,                      0.0],
        [L_SHANK * np.sin(k) - tx, -tx],
        [0.0,                      0.0],
        [1.0,                      1.0],
        [0.0,                      0.0],
    ])


def Ad_inv_gft(a, k):
    """Ad(g_ft)^-1 == Ad(g_ft^-1), 6x6, maps a foot-frame twist to thigh frame.
    g_ft = (R_ft, p_ft) with R_ft = rot_y(a+k), p_ft = fk_p_fo_th (offset incl).
    Ad(g^-1) = [[R^T, hat(-R^T p) @ R^T], [0, R^T]]."""
    R = _rot_y(a + k)
    p = fk_p_fo_th(a, k)
    Rt = R.T
    top = np.hstack([Rt, hat(-Rt @ p) @ Rt])
    bot = np.hstack([np.zeros((3, 3)), Rt])
    return np.vstack([top, bot])


# ---------------------------------------------------------------------------
# Measurements. Each returns (r, V): r (3,), V (3,3). H is the implicit 3x3
# identity selector at the block noted in each function; see eskf._kalman_update.
# ---------------------------------------------------------------------------
def meas_zupt(fo, sigma_zupt):
    """Zero-velocity update on the foot. h(x) = v_fo, y = 0.

    Only call when the stationarity detector reports the foot still (which is
    a subset of stance, not the same thing). This is the
    absolute anchor -- it corrects the foot directly (H on dv_fo), and pulls
    foot tilt + accel bias through P's cross-covariance.

    Returns (r, V). H is the constant selector picking FO["dv"] (a 3x3 identity
    at cols 24:27); _kalman_update keys off the block start directly instead of
    forming the 3x30, so it is not returned. See FO_DV in eskf.py.
    """
    r = -fo.v                       # y - h = 0 - v_fo
    V = (sigma_zupt ** 2) * np.eye(3)
    return r, V


def meas_pos_loop_closure(fo, th, enc, P, R_enc, R_model):
    """Position loop closure, CASCADE form: h(x) = p_th, correct thigh only.

        y = p_fo + R_fo @ p_fo_th(enc)
        r = y - p_th

    Foot pose is treated as a pseudo-measurement, so its uncertainty must enter
    V (H_fo @ P_foot @ H_fo^T), alongside encoder noise mapped through J_pos.

    R_enc  : 2x2 cov of [ankle_angle, knee_angle] noise.
    R_model: 3x3 floor for unmodeled compliance (ankle spring etc.).
    """
    a, k = _chain_angles(enc)
    p_ft = fk_p_fo_th(a, k)
    R_fo = q_to_R(fo.q)

    y = fo.p + R_fo @ p_ft
    r = y - th.p

    # H is the constant selector picking TH["dp"] (3x3 identity at cols 6:9);
    # _kalman_update keys off the block start (TH_DP) rather than the 3x30.

    # foot-pose uncertainty: d y = dp_fo - R_fo hat(p_ft) dtheta_fo
    H_dp = np.eye(3)
    H_dth = -R_fo @ hat(p_ft)
    P_dp = P[FO["dp"], FO["dp"]]
    P_dth = P[FO["dtheta"], FO["dtheta"]]
    P_cross = P[FO["dp"], FO["dtheta"]]
    V_foot = (H_dp @ P_dp @ H_dp.T
              + H_dth @ P_dth @ H_dth.T
              + H_dp @ P_cross @ H_dth.T
              + H_dth @ P_cross.T @ H_dp.T)

    # encoder noise: chain angle map is linear so S cancels in covariance;
    # J_pos is in chain convention, R_enc in encoder convention -- same 2x2.
    J = pos_jacobian(a, k)
    V_enc = R_fo @ J @ R_enc @ J.T @ R_fo.T

    V = V_foot + V_enc + R_model
    return r, V


def meas_vel_loop_closure(fo, th, enc, theta_dot_enc, foot_gyro, P,
                          R_rate, R_model):
    """Velocity loop closure, CASCADE form: h(x) = v_th, correct thigh only.

    BODY-TWIST form (MLS). Compose the foot's body twist with the leg body
    Jacobian, then rotate the thigh body twist's linear part back to nav:

        V_b_nf = [R_fo.T @ v_fo ; omega_fo]        (foot body twist)
        V_b_nt = Ad(g_ft)^-1 @ V_b_nf + J_b @ theta_dot
        v_th   = R_th @ V_b_nt[:3]                 (thigh origin velocity, nav)
        r      = v_th - th.v

    This avoids the nav-frame omega x p transport term (that is an artifact of
    the spatial form; the body form has none -- v_b == R.T @ p_dot). omega_fo is
    the bias-corrected foot gyro, i.e. the foot body angular velocity directly.
    theta_dot is CHAIN convention, so apply the S sign flips to encoder rates.

    theta_dot_enc : (2,) [ankle_rate, knee_rate] from encoder derivatives.
    foot_gyro     : (3,) raw foot gyro this tick.
    R_rate        : 2x2 cov of joint-rate noise (raw discrete-derivative -> big).
    R_model       : 3x3 model/compliance floor.

    NOTE singular at k=0 (straight knee): J_b loses a direction. Let V reflect
    it; do not special-case.
    """
    a, k = _chain_angles(enc)
    theta_dot = np.array([S_ANKLE, S_KNEE]) * np.asarray(theta_dot_enc)

    R_fo = q_to_R(fo.q)
    R_th = q_to_R(th.q)
    omega_fo = np.asarray(foot_gyro) - fo.w_b        # foot body angular velocity
    Ai = Ad_inv_gft(a, k)
    J_b = body_jacobian(a, k)

    V_b_nf = np.concatenate([R_fo.T @ fo.v, omega_fo])
    V_b_nt = Ai @ V_b_nf + J_b @ theta_dot
    y = R_th @ V_b_nt[:3]
    r = y - th.v

    # H is the constant selector picking TH["dv"] (3x3 identity at cols 9:12);
    # _kalman_update keys off the block start (TH_DV) rather than the 3x30.

    # foot-side states treated as measurement -> push into V. Linear-velocity
    # map is v_th_pred = R_th @ Ai[:3] @ V_b_nf; differentiate the foot blocks:
    #   d/dv_fo   = R_th @ Ai_ll @ R_fo.T
    #   d/dw_b_fo = -R_th @ Ai_la          (omega_fo = gyro - w_b)
    # rate:  d/theta_dot = R_th @ J_b[:3];  S flips square out in covariance.
    Ai_ll = Ai[:3, :3]
    Ai_la = Ai[:3, 3:6]
    H_vfo = R_th @ Ai_ll @ R_fo.T
    H_wb = -R_th @ Ai_la
    J_lin = R_th @ J_b[:3, :]
    P_dv = P[FO["dv"], FO["dv"]]
    P_dwb = P[FO["dw_b"], FO["dw_b"]]
    V_foot = H_vfo @ P_dv @ H_vfo.T + H_wb @ P_dwb @ H_wb.T   # attitude terms dropped (draft)

    V_rate = J_lin @ R_rate @ J_lin.T
    V = V_foot + V_rate + R_model

    # Thigh-attitude Jacobian block for H (the term the cascade draft dropped).
    # y = R_th @ V_b_nt[:3], and under the local/right error R_true=R_nom(I+hat)
    # dh/d(dtheta_th) = +R_th @ hat(V_b_nt[:3]) in the same (+I on dv) sign family
    # as this closure's velocity block. Numerically verified vs finite diff of y.
    # Caller decides whether to use it (cfg.correct_thigh_attitude).
    H_dth_th = R_th @ hat(V_b_nt[:3])
    return r, V, H_dth_th
