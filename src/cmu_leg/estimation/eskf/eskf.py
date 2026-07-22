"""
Error State Kalman Filter for Kinematic State Estimation

For general introduction of Error State Kalman Filter in IMU 
navigation problems consult Joan Sola, "Quaternion kinematics for 
the error-state Kalman filter" (2017, arXiv:1711.02508v1)

Follows Haosen (Russell) Xing's implementation of ESKF.

Usage:
    class StateEstimator:
        def __init__(self):
            self._eskf = ESKF()            # or ESKF(cfg) to override ESKF_CFG

        def get_state(self, frame, imu, t):
            self._apply_encoders(frame)    # must precede update(), see below
            self._eskf.update(imu, self.state, t, dt)   # writes hip_* itself
            return self.state

update() owns the hip fields and writes them onto the bus directly, so the
caller never has to reach into the filter's internals. Check .initialized
before trusting them.

The encoder conversion has to run first: the loop closures read the joint
angles and velocities off `ProsthesisState`, so calling update() before them fuses the
previous tick's leg shape.

t is the caller's clock in seconds and must share the IMU driver's
perf_counter timebase. Call update() every tick and in order -- it integrates
t - t_prev, so a skipped tick is a skipped dt. The first cfg.init_duration
seconds are complementary-filter alignment, so treat the estimates as invalid
until then, and check .healthy for a stalled IMU stream.

Author: vkalva
"""

from cmu_leg.estimation.eskf.helpers import q_to_R, S, hat, norm, Euler_angles_collocating_vectors, quat_prod, R_to_Euler, angle_axis_to_q ,NominalState
from cmu_leg.estimation.eskf.measurements import meas_pos_loop_closure, meas_vel_loop_closure, meas_zupt
from cmu_leg.estimation.eskf.detectors import ZUPT_AdaptiveGLRT
from cmu_leg.config.params import ESKFConfig, ESKF_CFG, LEG
from cmu_leg.types.sensors import IMUFrame
from cmu_leg.types.state import ProsthesisState
import numpy as np

# Sensor -> body maps. The permutation parts are exact 90 deg axis swaps. The
# thigh additionally carries the physical mount tilt (LEG.thigh_imu_pitch), so
# the thigh BODY frame is link-aligned and the chain's
# R_fo.T @ R_th == rot_y(a + k) identity holds without a residual constant.
# measurements.py applies the SAME rot_y to knee_to_thigh_imu -- keep them in
# step, they are two effects of one physical angle. The foot IMU is mounted
# flush (confirmed), so it gets the permutation only.
_PERM_TH = np.array([[0.0, -1.0, 0.0],
                     [1.0,  0.0, 0.0],
                     [0.0,  0.0, 1.0]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


R_BODY_SENSOR_TH = _rot_y(-LEG.thigh_imu_pitch) @ _PERM_TH
R_BODY_SENSOR_FO = np.array([[-1.0, 0.0, 0.0],
                             [ 0.0, 0.0, 1.0],
                             [ 0.0, 1.0, 0.0]])


class ESKF:
    def __init__(self, cfg: ESKFConfig | None = None):
        self.cfg = cfg = cfg or ESKF_CFG
        self.th = NominalState()
        self.fo = NominalState()
        self.g = np.array([0, 0, -9.81])

        p0_body = ([cfg.p0_att] * 3 + [cfg.p0_gyr_bias] * 3 + [cfg.p0_pos] * 3
                   + [cfg.p0_vel] * 3 + [cfg.p0_acc_bias] * 3)
        self.P = np.diag(p0_body * 2)
        self.del_x = np.zeros((30,))

        self._sig_th = (cfg.sigma_dtheta_thigh, cfg.sigma_gyr_thigh,
                        cfg.sigma_dv_thigh, cfg.sigma_acc_thigh)
        self._sig_fo = (cfg.sigma_dtheta_foot, cfg.sigma_gyr_foot,
                        cfg.sigma_dv_foot, cfg.sigma_acc_foot)
        self._prev_corr_th = (np.zeros(3), np.zeros(3))   # (acc, gyr)
        self._prev_corr_fo = (np.zeros(3), np.zeros(3))

        # TODO: params to cfg
        self._zupt_fo = ZUPT_AdaptiveGLRT(
            hist_len=30, adapt_hist_len=1000,
            sigma_acc=0.05, sigma_gyr=0.01, min_offset=0.1)
        self._zupt_active = False

        self.R_enc = np.eye(2) * cfg.sigma_enc ** 2
        self.R_rate = np.eye(2) * cfg.sigma_rate ** 2
        self.R_model_pos = np.eye(3) * cfg.sigma_model_pos ** 2
        self.R_model_vel = np.eye(3) * cfg.sigma_model_vel ** 2
        self.sigma_zupt = cfg.sigma_zupt

        self.init_duration = cfg.init_duration
        self.beta = cfg.comp_beta
        self.t_init = None
        self.t_prev = None
        self.healthy = True

    def update(self, imu: IMUFrame, state: ProsthesisState, t, dt):
        """
        Advance the filter one tick: comp-filter init for the first
        cfg.init_duration seconds, then predict + correct.

        Inputs
            imu: IMUFrame written by hardware interface
            state: Prosthesis as returned by the state estimator
            t: wall-clock (perf_counter) seconds, SAME timebase as the IMU
               driver -- the staleness check below compares them directly
            dt: measured elapsed since the previous tick, clamped by the caller

        Updates self.th / self.fo / self.P in place and sets self.healthy.
        Runs on the caller's clock, holding the last IMU sample between the
        ~200 Hz stream and the faster loop.

        `state` is IN/OUT. The loop closures READ knee/ankle angle and velocity
        off it, and _write_outputs WRITES the hip fields back. Disjoint fields,
        so nothing clobbers, but both directions need convert_encoders to have
        run first -- see the module docstring.
        """
        # sensor frame -> body frame
        a_m_th, w_m_th, a_m_fo, w_m_fo = self._sensor_to_body(imu)

        # stream health. Only meaningful because t and the IMU timestamps share
        # a perf_counter timebase -- an idealised loop clock would drift against
        # them and eventually report a healthy IMU as stale.
        self.healthy = (t - imu.thigh_time <= self.cfg.imu_stale_timeout
                        and t - imu.foot_time <= self.cfg.imu_stale_timeout)
        self.t_prev = t

        # first call: latch the init clock, seed both attitudes to identity
        if self.th.q is None or self.fo.q is None:
            self.t_init = t
            self.th.q = np.array([1.0, 0.0, 0.0, 0.0])
            self.fo.q = np.array([1.0, 0.0, 0.0, 0.0])

        if t < self.t_init + self.init_duration:
            # init window: align attitude to gravity, warm the detector
            self.th.q = self.ori_comp_filter(q_prev=self.th.q, gyr=w_m_th, acc=a_m_th, g_n=self.g, beta=self.beta, dt=dt)
            self.fo.q = self.ori_comp_filter(q_prev=self.fo.q, gyr=w_m_fo, acc=a_m_fo, g_n=self.g, beta=self.beta, dt=dt)
            if imu.foot_fresh:
                self._zupt_fo.update(a_m_fo, w_m_fo)

        else:
            self._predict(a_m_th, w_m_th, a_m_fo, w_m_fo, dt)
            self._correct(a_m_fo, w_m_fo, imu.foot_fresh, state)

        # publish onto the control bus -- last, so it sees the corrected state
        self._write_outputs(state, w_m_th)

    @property
    def initialized(self) -> bool:
        """True once the comp-filter alignment window has closed. Until then
        the attitudes are still converging and the biases are untouched, so
        nothing downstream should trust the outputs."""
        return (self.t_init is not None
                and self.t_prev >= self.t_init + self.init_duration)

    def _write_outputs(self, state: ProsthesisState, w_m_th):
        """Write the ESKF-owned fields onto the control bus.

        hip_angle is the thigh angle from vertical, flexion positive

            hip_angle = -(thigh link pitch)

        The thigh BODY frame is link-aligned -- LEG.thigh_imu_pitch is folded
        into R_BODY_SENSOR_TH -- so th.q's pitch IS the link pitch, no extra
        constant. Sign agrees with the encoders' flexion-positive convention,
        which is what lets the baseline write knee_desired = hip_angle - offset.

        hip_velocity comes off the gyro

        hip_accel is deliberately not written: it needs a differentiator, and
        StateEstimator already owns all of those.
        """
        if not self.initialized:
            # leave the previous values on the bus rather than publishing an
            # unconverged attitude
            return

        roll, pitch, _ = R_to_Euler(q_to_R(self.th.q))
        w_th = w_m_th - self.th.w_b            # bias-corrected, thigh body frame

        state.hip_angle = -pitch
        state.hip_velocity = -(np.cos(roll) * w_th[1] - np.sin(roll) * w_th[2])

    @staticmethod
    def _sensor_to_body(imu: IMUFrame):
        return (R_BODY_SENSOR_TH @ np.asarray(imu.thigh_acc),
                R_BODY_SENSOR_TH @ np.asarray(imu.thigh_gyro),
                R_BODY_SENSOR_FO @ np.asarray(imu.foot_acc),
                R_BODY_SENSOR_FO @ np.asarray(imu.foot_gyro))

    def _predict(self, a_m_th, w_m_th, a_m_fo, w_m_fo, dt: float):
        # integrate the nominal thigh state (nominal biases stay constant)
        w_corr_th = w_m_th - self.th.w_b
        self.th.q = quat_prod(self.th.q, angle_axis_to_q(w_corr_th * dt))
        self.th.q = self.th.q / norm(self.th.q)
        R_th = q_to_R(self.th.q)
        a_corr_th = a_m_th - self.th.a_b
        a_n_th = R_th @ a_corr_th
        self.th.p = self.th.p + self.th.v * dt + 0.5 * (a_n_th + self.g) * dt**2
        self.th.v = self.th.v + (a_n_th + self.g) * dt

        # integrate the nominal foot state
        w_corr_fo = w_m_fo - self.fo.w_b
        self.fo.q = quat_prod(self.fo.q, angle_axis_to_q(w_corr_fo * dt))
        self.fo.q = self.fo.q / norm(self.fo.q)
        R_fo = q_to_R(self.fo.q)
        a_corr_fo = a_m_fo - self.fo.a_b
        a_n_fo = R_fo @ a_corr_fo
        self.fo.p = self.fo.p + self.fo.v * dt + 0.5 * (a_n_fo + self.g) * dt**2
        self.fo.v = self.fo.v + (a_n_fo + self.g) * dt

        # error-state transition and process noise, one block per body
        A = np.zeros((30, 30))
        A[0:15, 0:15] = self._get_A(R_th, a_n_th, dt)
        A[15:30, 15:30] = self._get_A(R_fo, a_n_fo, dt)
        Q = np.zeros((30, 30))
        Q[0:15, 0:15] = self._get_Q(self._sig_th, R_th, a_corr_th, w_corr_th,
                                    self._prev_corr_th, dt)
        Q[15:30, 15:30] = self._get_Q(self._sig_fo, R_fo, a_corr_fo, w_corr_fo,
                                      self._prev_corr_fo, dt)
        self._prev_corr_th = (a_corr_th, w_corr_th)
        self._prev_corr_fo = (a_corr_fo, w_corr_fo)

        # propagate covariance (del_x itself is always zero here, so it is not
        # propagated: every correction injects and resets it)
        # del_x = A @ del_x 
        self.P = A @ self.P @ A.T + Q

    def _get_Q(self, sig, R_nb, acc_corr, gyr_corr, prev_corr, dt):
        # white-noise variances over one step
        sigma_dtheta, sigma_gyr, sigma_dv, sigma_acc = sig
        a = (sigma_dtheta * dt) ** 2
        b = (sigma_gyr * dt) ** 2
        c = (sigma_dv * dt) ** 2
        d = (sigma_acc * dt) ** 2

        if self.cfg.dynamic_process_noise:
            # inflate attitude/velocity by this tick's integration error
            prev_acc, prev_gyr = prev_corr
            aa = (R_nb @ (gyr_corr - prev_gyr) / 2 * dt) ** 2 + a
            cc = (R_nb @ (acc_corr - prev_acc) / 2 * dt) ** 2 + c
            return np.diag([aa[0], aa[1], aa[2], b, b, b, 0, 0, 0,
                            cc[0], cc[1], cc[2], d, d, d])

        return np.diag([a, a, a, b, b, b, 0, 0, 0, c, c, c, d, d, d])

    @staticmethod
    def _get_A(R_nb, a_n, dt):
        # LOCAL 
        A = np.eye(15)
        A[0:3, 3:6] = -dt * np.eye(3)          # ori  <- gyro bias
        A[6:9, 9:12] = dt * np.eye(3)          # pos  <- vel
        A[9:12, 0:3] = -dt * hat(a_n) @ R_nb   # vel  <- ori
        A[9:12, 12:15] = -dt * R_nb            # vel  <- accel bias
        return A

    def _correct(self, a_m_fo, w_m_fo, fo_fresh: bool, state: ProsthesisState):
        theta_dot = np.array([state.ankle_velocity, state.knee_velocity])

        # Detector only sees fresh samples (held ones would fake-shrink its
        # window variance); the correction reads fo.v only, so it can run every
        # tick on the held decision.
        if fo_fresh:
            self._zupt_active = self._zupt_fo.update(a_m_fo, w_m_fo)
        if self._zupt_active:
            r, H, V = meas_zupt(self.fo, self.sigma_zupt)
            self._kalman_update(r, H, V)

        # encoders tie the thigh to the foot: position, then velocity
        r, H, V = meas_pos_loop_closure(
            self.fo, self.th, state, self.P, self.R_enc, self.R_model_pos)
        self._kalman_update(r, H, V)

        r, H, V = meas_vel_loop_closure(
            self.fo, self.th, state, theta_dot, w_m_fo, self.P,
            self.R_rate, self.R_model_vel)
        self._kalman_update(r, H, V)

    def _kalman_update(self, r, H, V):
        # gain, error estimate, covariance update (Joseph form)
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + V)
        self.del_x = K @ r
        self.P = (np.eye(30) - K @ H) @ self.P @ (np.eye(30) - K @ H).T + K @ V @ K.T

        # inject the error into each nominal state
        for body, offset in ((self.th, 0), (self.fo, 15)):
            dtheta = self.del_x[offset + 0 : offset + 3]
            dq = np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])
            body.q = quat_prod(body.q, dq)
            body.q = body.q / norm(body.q)
            body.w_b = body.w_b + self.del_x[offset + 3 : offset + 6]
            body.p = body.p + self.del_x[offset + 6 : offset + 9]
            body.v = body.v + self.del_x[offset + 9 : offset + 12]
            body.a_b = body.a_b + self.del_x[offset + 12 : offset + 15]

        # reset the error state
        self.del_x = np.zeros((30,))

    def ori_comp_filter(self, q_prev, gyr, acc, g_n, dt, beta):
        """
        Estimate orientation quaternion q at time t between navigation (n) 
        and IMU frame (b) using high-pass (for w) and low-pass 
        (for acc) complementary filter with one-step gradient descent 
        for the orientation estimate coming from acc.

        Input: q_prev: orientation quaternion estimate of previous 
                    time step, format: q = [q0, qx, qy, qz]
            gyr: current time gyro measurement, format: [wx, wy, wz]
            acc: current time accelerometer measurement: [ax, ay, az]

            g_n: gravity vector in navigation frame: [gx, gy, gz]

            dt: time interval between current sensor readings
                and previous quaternion estimate, dt = t - t_prev 
            beta: step size of normalized gradient descent 
                    (filter tuning parameter; reasonable choice is 
                    beta = sqrt(3) * sigma, with sigma being the standard 
                    deviation of the gyroscope noise; based on standard 
                    deviation of integration drift for unit quaternion: 
                    sqrt(3) * sigma * dt)

        Output: q: orientation quaternion estimate at time t:
                [q0, qx, qy, qz], norm(q) = 1

        Implementation follows paper
        'A fast and robust algorithm for orientation estimation using
        inertial sensors' by M. Kok and T. B. Schoen in IEEE Signal
        Processing Letters, 2019.
        """
        # a dropped (zero) accel sample skips the correction, keeps the gyro
        w_est = np.asarray(gyr, dtype=float)
        if norm(acc) > 0:
            # gradient of the gravity-misalignment cost (eq 12b)
            acc = acc / norm(acc)
            g_n = g_n / norm(g_n)
            g_b = (q_to_R(q_prev)).T @ g_n
            grad_V = -hat(g_b) @ (acc + g_b)
            # corrected rate: gyro minus a fixed-rate pull downhill (eq 15b)
            if norm(grad_V) > 1e-8:
                w_est = w_est - beta * grad_V / norm(grad_V)

        # integrate (eq 15a)
        q = q_prev + dt / 2 * S(q_prev) @ w_est
        return q / norm(q)
