
def jacobian_A(in1, in2):

    R = in2[9]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    r = in2[8]
    th = in1[2]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R * psiA * r
    t5 = l + l0 + t4

    # Create out1 as a single row vector
    out1 = np.array([1.0,0.0,0.0,1.0,t2*t5,t3*t5,R*r*t3,-R*r*t2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    # Reshape to 2x9 matrix
    return out1.reshape((2, 9), order='F')


def jacobian_B(in1, in2):

    R = in2[9]
    l = in2[3]
    l0 = in2[7]
    psiB = in1[4]
    r = in2[8]
    th = in1[2]

    t2 = R * psiB * r
    t3 = np.pi / 3.0
    t4 = t3 + th
    t5 = l + l0 + t2
    t6 = np.cos(t4)
    t7 = np.sin(t4)

    out1 = np.array([
        [1.0, 0.0, 0.0, 1.0, t5 * t6, t5 * t7, 0.0, 0.0, R * r * t7, -R * r * t6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    return out1.reshape((2, 9), order='F')


def jacobian_C(in1, in2):

    R = in2[9]
    l = in2[3]
    l0 = in2[7]
    psiC = in1[5]
    r = in2[8]
    th = in1[2]

    t2 = R * psiC * r
    t3 = np.pi * (2.0 / 3.0)
    t4 = t3 + th
    t5 = l + l0 + t2
    t6 = np.cos(t4)
    t7 = np.sin(t4)

    out1 = np.array([
        [1.0, 0.0, 0.0, 1.0, t5 * t6, t5 * t7, 0.0, 0.0, 0.0, 0.0, 0.0, R * r * t7, -R * r * t6, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    return out1.reshape((2, 9), order='F')


def jacobian_D(in1, in2):

    R = in2[9]
    l = in2[3]
    l0 = in2[7]
    psiD = in1[6]
    r = in2[8]
    th = in1[2]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R * psiD * r
    t5 = l + l0 + t4

    out1 = np.array([
        [1.0, 0.0, 0.0, 1.0, -t2 * t5, -t3 * t5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -R * r * t3, R * r * t2, 0.0, 0.0, 0.0]
    ])

    return out1.reshape((2, 9), order='F')


def jacobian_E(in1, in2):

    R = in2[9]
    l = in2[3]
    l0 = in2[7]
    psiE = in1[7]
    r = in2[8]
    th = in1[2]

    t2 = R * psiE * r
    t3 = np.pi * (4.0 / 3.0)
    t4 = t3 + th
    t5 = l + l0 + t2
    t6 = np.cos(t4)
    t7 = np.sin(t4)

    out1 = np.array([
        [1.0, 0.0, 0.0, 1.0, t5 * t6, t5 * t7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R * r * t7, -R * r * t6, 0.0]
    ])

    return out1.reshape((2, 9), order='F')


def jacobian_F(in1, in2):

    R = in2[9]
    l = in2[3]
    l0 = in2[7]
    psiF = in1[8]
    r = in2[8]
    th = in1[2]

    t2 = R * psiF * r
    t3 = np.pi * (5.0 / 3.0)
    t4 = t3 + th
    t5 = l + l0 + t2
    t6 = np.cos(t4)
    t7 = np.sin(t4)

    out1 = np.array([
        [1.0, 0.0, 0.0, 1.0, t5 * t6, t5 * t7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R * r * t7, -R * r * t6]
    ])

    return out1.reshape((2, 9), order='F')


def A_rimless_wheel(in1, in2):

    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]

    t2 = R ** 2
    t3 = Il * t2
    t4 = Im + t3

    A = np.zeros((9, 9))
    A[0, 0] = M
    A[1, 1] = M
    A[2, 2] = I
    A[3:, 3:] = np.eye(6) * t4

    return A


def b_rimless_wheel(in1, in2):

    M = in2[0]
    g = in2[4]
    ramp = in2[5]

    b = np.array([
        M * g * np.sin(ramp),
        -M * g * np.cos(ramp),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])

    return b


def keypoints_spokes_rimless_wheel(in1, in2):

    l = in2[3]
    th = in1[2]
    x = in1[0]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t6 = np.pi / 3.0
    t7 = 2.0 * np.pi / 3.0
    t8 = 4.0 * np.pi / 3.0
    t9 = 5.0 * np.pi / 3.0

    t4 = l * t2
    t5 = l * t3
    t10 = t6 + th
    t11 = t7 + th
    t12 = t8 + th
    t13 = t9 + th

    keypoints_spokes = np.array([
        [t5 + x, -t4 + y],
        [x + l * np.sin(t10), y - l * np.cos(t10)],
        [x + l * np.sin(t11), y - l * np.cos(t11)],
        [-t5 + x, t4 + y],
        [x + l * np.sin(t12), y - l * np.cos(t12)],
        [x + l * np.sin(t13), y - l * np.cos(t13)],
    ]).T

    return keypoints_spokes


def keypoints_vel_rimless_wheel(in1, in2):
    R = in2[9]
    dpsiA, dpsiB, dpsiC, dpsiD, dpsiE, dpsiF = in1[12:18]
    dth = in1[11]
    dx, dy = in1[9], in1[10]
    l = in2[3]
    l0 = in2[7]
    psiA, psiB, psiC, psiD, psiE, psiF = in1[3:9]
    r = in2[8]
    th = in1[2]

    t2, t3 = np.cos(th), np.sin(th)
    t4, t5, t6, t7, t8, t9 = R * psiA * r, R * psiB * r, R * psiC * r, R * psiD * r, R * psiE * r, R * psiF * r
    t10, t11, t12, t13 = np.pi / 3.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 5.0 * np.pi / 3.0
    t14, t15, t16, t17 = t10 + th, t11 + th, t12 + th, t13 + th
    t18, t19, t20, t21, t22, t23 = l + l0 + t4, l + l0 + t5, l + l0 + t6, l + l0 + t7, l + l0 + t8, l + l0 + t9
    t24, t25, t26, t27 = np.cos(t14), np.cos(t15), np.cos(t16), np.cos(t17)
    t28, t29, t30, t31 = np.sin(t14), np.sin(t15), np.sin(t16), np.sin(t17)

    keypoints_vel = np.array([
        [dx + dth * t2 * t18 + R * dpsiA * r * t3, dy + dth * t3 * t18 - R * dpsiA * r * t2],
        [dx + dth * t19 * t24 + R * dpsiB * r * t28, dy + dth * t19 * t28 - R * dpsiB * r * t24],
        [dx + dth * t20 * t25 + R * dpsiC * r * t29, dy + dth * t20 * t29 - R * dpsiC * r * t25],
        [dx - dth * t2 * t21 - R * dpsiD * r * t3, dy - dth * t3 * t21 + R * dpsiD * r * t2],
        [dx + dth * t22 * t26 + R * dpsiE * r * t30, dy + dth * t22 * t30 - R * dpsiE * r * t26],
        [dx + dth * t23 * t27 + R * dpsiF * r * t31, dy + dth * t23 * t31 - R * dpsiF * r * t27]
    ]).T

    return keypoints_vel

import numpy as np

def keypoints_rimless_wheel(in1, in2):

    R = in2[9]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    psiB = in1[4]
    psiC = in1[5]
    psiD = in1[6]
    psiE = in1[7]
    psiF = in1[8]
    r = in2[8]
    th = in1[2]
    x = in1[0]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R * psiA * r
    t5 = R * psiB * r
    t6 = R * psiC * r
    t7 = R * psiD * r
    t8 = R * psiE * r
    t9 = R * psiF * r
    t10 = np.pi / 3.0
    t11 = np.pi * (2.0 / 3.0)
    t12 = np.pi * (4.0 / 3.0)
    t13 = np.pi * (5.0 / 3.0)
    t14 = t10 + th
    t15 = t11 + th
    t16 = t12 + th
    t17 = t13 + th
    t18 = l + l0 + t4
    t19 = l + l0 + t5
    t20 = l + l0 + t6
    t21 = l + l0 + t7
    t22 = l + l0 + t8
    t23 = l + l0 + t9

    keypoints = np.array([
        [x + t3 * t18, y - t2 * t18],
        [x + t19 * np.sin(t14), y - t19 * np.cos(t14)],
        [x + t20 * np.sin(t15), y - t20 * np.cos(t15)],
        [x - t3 * t21, y + t2 * t21],
        [x + t22 * np.sin(t16), y - t22 * np.cos(t16)],
        [x + t23 * np.sin(t17), y - t23 * np.cos(t17)]
    ])

    return keypoints.T

def keypoints_spokes_rimless_wheel3(in1, in2):
    l = in2[3]
    th = in1[2]
    x = in1[0]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t6 = np.pi / 3.0
    t7 = 2.0 * np.pi / 3.0
    t8 = 4.0 * np.pi / 3.0
    t9 = 5.0 * np.pi / 3.0

    t4 = l * t2
    t5 = l * t3
    t10 = t6 + th
    t11 = t7 + th
    t12 = t8 + th
    t13 = t9 + th

    keypoints_spokes3 = np.array([
        [t5 + x, -t4 + y],
        [x + l * np.sin(t10), y - l * np.cos(t10)],
        [x + l * np.sin(t11), y - l * np.cos(t11)],
        [-t5 + x, t4 + y],
        [x + l * np.sin(t12), y - l * np.cos(t12)],
        [x + l * np.sin(t13), y - l * np.cos(t13)],
    ]).T

    return keypoints_spokes3

