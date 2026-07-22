#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass, field

#####################################################################
# dataclasses for ESKF
#####################################################################

@dataclass
class NominalState:
    q: np.ndarray = None #field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # body->nav quaternion [q0, qx, qy, qz]
    w_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))      # gyroscope bias [rad/s]
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))        # position, nav frame [m]
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))        # velocity, nav frame [m/s]
    a_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))      # accelerometer bias [m/s^2]




def norm(x):
    """Compute norm of a vector"""
    return np.sqrt(np.dot(x, x))


######################################################################
def hat(a):
    """Matrix representation of cross product  a x b = hat(a) * b."""
    a1 = a[0];  a2 = a[1];  a3 = a[2]

    return np.array([[0,  -a3,  a2],
                     [a3,   0, -a1],
                     [-a2, a1,   0]])


######################################################################
def quat_prod(p, q):
    """Quaternion product operator p o q."""
    p0, px, py, pz = p[0], p[1], p[2], p[3]
    q0, qx, qy, qz = q[0], q[1], q[2], q[3]

    return np.array([p0 * q0 - px * qx - py * qy - pz * qz,
                     p0 * qx + px * q0 + py * qz - pz * qy,
                     p0 * qy - px * qz + py * q0 + pz * qx,
                     p0 * qz + px * qy - py * qx + pz * q0])


######################################################################
def S(p):
    """Quaternion product operator for product p o q = S(p) * q with 
    a pure quaternion q (scalar term of q is zero)."""
    p0 = p[0]; px = p[1]; py = p[2]; pz = p[3]

    return np.array([[-px, -py, -pz],
                     [p0,  -pz,  py],
                     [pz,   p0, -px],
                     [-py,  px,  p0]])


######################################################################
def q_to_R(q):
    """Convert quaternion [q0, q_x, q_y, q_z] to 3x3 rotation matrix
    with format [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]].
    """
    q0 = q[0]; qx = q[1]; qy = q[2]; qz = q[3]

    r11 = 1 - 2 * (qy * qy + qz * qz)
    r12 = 2 * (qx * qy - qz * q0)
    r13 = 2 * (qx * qz + qy * q0)

    r21 = 2 * (qx * qy + qz * q0)
    r22 = 1 - 2 * (qx * qx + qz * qz)
    r23 = 2 * (qy * qz - qx * q0)

    r31 = 2 * (qx * qz - qy * q0)
    r32 = 2 * (qy * qz + qx * q0)
    r33 = 1 - 2 * (qx * qx + qy * qy)

    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


######################################################################
def angle_axis_to_R(angle, axis):
    """Convert angle-axis representation of rotation to rotation matrix R
    using Rodrigues' formula."""
    ax_hat = hat(axis)
    R = np.eye(3) + np.sin(angle) * ax_hat \
        + (1 - np.cos(angle)) * ax_hat @ ax_hat

    return R


######################################################################
def R_to_Euler(R):
    """ Convert 3x3 rotation matrix R_ab with format [[r11, r12, r13], 
    [r21, r22, r23], [r31, r32, r33]] to roll, pitch and yaw angles 
    assuming R_ab = R_z(yaw) * R_y(pitch) * R_x(roll)."""
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


######################################################################


def R_to_q(R, q0_sign=1):
    """Convert rotation matrix to quaternion based on method described
    in Sarabandi & Thomas, Accurate Computation of Quaternions from 
    Rotation Matrices, in Intl Symposium on Advances in Robot
    Kinematics, 2018.

    Input: rotation matrix R = [[r11, r12, r13], 
                                [r21, r22, r23],
                                [r31, r32, r33]]

           q0_sign: Desired sign of q0: 0 or 1 means positive, 
                    -1 means negative. (Conversion from R to q is
                    not unique (there are four solutions; typically
                    either of two is assigned, based on desired sign
                    of q0).  

    Output: quaternion q = [q0, qx, qy, qz]]"""

    THRESHOLD = 0.0

    # extract elements of R
    r11 = R[0, 0];  r12 = R[0, 1];  r13 = R[0, 2]
    r21 = R[1, 0];  r22 = R[1, 1];  r23 = R[1, 2]
    r31 = R[2, 0];  r32 = R[2, 1];  r33 = R[2, 2]

    r32_23_sq = (r32 - r23) * (r32 - r23)
    r13_31_sq = (r13 - r31) * (r13 - r31)
    r21_12_sq = (r21 - r12) * (r21 - r12)
    r1221_sq = (r12 + r21) * (r12 + r21)
    r3113_sq = (r31 + r13) * (r31 + r13)
    r2332_sq = (r23 + r32) * (r23 + r32)

    # assign q0 (eq. 23 in cited paper)
    r112233 = r11 + r22 + r33
    if r112233 > THRESHOLD:
        q0 = 0.5 * np.sqrt(1 + r112233)
    else:
        q0 = 0.5 * np.sqrt((r32_23_sq + r13_31_sq + r21_12_sq)
                           / (3 - r112233))

    # assign q1 (eq. 24)
    r11_22_33 = r11 - r22 - r33
    if r11_22_33 > THRESHOLD:
        qx = 0.5 * np.sqrt(1 + r11_22_33)
    else:
        qx = 0.5 * np.sqrt((r32_23_sq + r1221_sq + r3113_sq)
                           / (3 - r11_22_33))

    # assign q2 (eq. 25)
    r_1122_33 = - r11 + r22 - r33
    if r_1122_33 > THRESHOLD:
        qy = 0.5 * np.sqrt(1 + r_1122_33)
    else:
        qy = 0.5 * np.sqrt((r13_31_sq + r1221_sq + r2332_sq)
                           / (3 - r_1122_33))

    # assign q3 (eq. 26)
    r_11_2233 = - r11 - r22 + r33
    if r_11_2233 > THRESHOLD:
        qz = 0.5 * np.sqrt(1 + r_11_2233)
    else:
        qz = 0.5 * np.sqrt((r21_12_sq + r3113_sq + r2332_sq)
                           / (3 - r_11_2233))

    # assign signs
    if q0_sign >= 0:
        if np.sign(r32 - r23) < 0:
            qx = -qx
        if np.sign(r13 - r31) < 0:
            qy = -qy
        if np.sign(r21 - r12) < 0:
            qz = -qz
    else:
        q0 = -q0
        if np.sign(r32 - r23) >= 0:
            qx = -qx
        if np.sign(r13 - r31) >= 0:
            qy = -qy
        if np.sign(r21 - r12) >= 0:
            qz = -qz

    return np.array([q0, qx, qy, qz])


######################################################################
def proper_rotation(R):
    """Turn matrix into proper rotation matrix (orthogonal & det=1)"""
    I_3 = np.eye(3)
    A = (R.T - R) / (1 + R.trace())
    R = inv(I_3 + A) @ (I_3 - A)     # Cayley transform

    return R


######################################################################
def Euler_angles_collocating_vectors(a, b):
    """Compute Euler angles that rotate vector a onto vector b with
    a = [ax, ay, az] and b=[bx, by, bz]"""
    a = a / norm(a);  b = b / norm(b)
    c = np.cross(a, b)
    axis = c / norm(c)
    angle = np.arccos(np.dot(a, b))

    R = angle_axis_to_R(angle, axis)
    roll, pitch, yaw = R_to_Euler(R)

    return roll, pitch, yaw


######################################################################
def angle_axis_to_q(a):
    """Convert angle axis representation or rotation given by vector
    a into quaternion

    The axis is undefined at a == 0, so take the small-angle limit there:
    sin(angle/2)/angle -> 1/2, giving q = [1, a/2]. Without this the divide
    returns NaN, and since this runs on _predict's hot path a single all-zero
    bias-corrected gyro sample (a still, quantized IMU) would poison the whole
    filter state permanently.
    """
    a = np.asarray(a, dtype=float)
    norm_a = norm(a)
    if norm_a < 1e-12:
        return np.array([1.0, 0.5 * a[0], 0.5 * a[1], 0.5 * a[2]])

    angle = norm_a
    axis = a / norm_a
    q0 = np.cos(0.5 * angle)
    qvec = axis * np.sin(0.5 * angle)

    return np.array([q0, qvec[0], qvec[1], qvec[2]])


######################################################################
def unwrap_angle(angle, angle_vel, previous_unwrapped_angle, dt):
    """
    Unwrap angle.

    Inputs: 
        angle - wrapped angle at time t [rad]
        angle_vel - angular velocity at time t [rad/s]
        previous_unwrapped_angle - unwrapped angle at time t - dt [rad/s]
        dt - prediction time horizon [s] (equals time step)

    Output:
        unwrapped angle [rad]
    """

    predicted_angle = previous_unwrapped_angle + angle_vel * dt

    # compute number of full rotations to unwrap
    if predicted_angle > angle:
        n = np.floor(0.5 + (predicted_angle - angle) / (2 * np.pi))
    else:
        n = np.ceil(-0.5 + (predicted_angle - angle) / (2 * np.pi))

    return angle + 2 * np.pi * n
