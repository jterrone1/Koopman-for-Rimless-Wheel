import numpy as np

def ddq_0(in1, in2, in3):
    """
    Compute the base accelerations for a rimless wheel system with no spokes on the floor.
    """
    g = in2[4]
    ramp = in2[5]
    R = in2[9]
    Il = in2[10]
    Im = in2[11]

    uA = in3[0]
    uB = in3[1]
    uF = in3[2]

    t2 = R ** 2
    t3 = Il * t2
    t4 = Im + t3
    t5 = 1.0 / t4

    ddq_0 = np.array([
        g * np.sin(ramp),
        -g * np.cos(ramp),
        0.0,
        t5 * uA,
        t5 * uB,
        t5 * uF
    ])

    return ddq_0.reshape(6, 1)

def ddq_0_u(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to control inputs u for a rimless wheel system with no spokes on the floor.
    """
    R = in2[9]
    Il = in2[10]
    Im = in2[11]

    t2 = R ** 2
    t3 = Il * t2
    t4 = Im + t3
    t5 = 1.0 / t4

    ddq_0_u = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [t5,  0.0, 0.0],
        [0.0, t5,  0.0],
        [0.0, 0.0, t5]
    ])

    return ddq_0_u

def ddq_0_q(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to q for a rimless wheel system with no spokes on the floor.
    """
    return np.zeros((6, 6))

def ddq_0_dq(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to dq for a rimless wheel system with no spokes on the floor.
    """
    return np.zeros((6, 6))


def ddq_A(in1, in2, in3):
    """
    Compute the base accelerations for a rimless wheel system with spoke A in contact with the floor.
    """
    # Unpack inputs
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    g = in2[4]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    r = in2[8]
    ramp = in2[5]
    th = in1[2]
    uA = in3[0]
    uB = in3[1]
    uF = in3[2]
    y = in1[1]

    # Intermediate variables
    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R ** 2
    t5 = R * psiA * r
    t6 = 1.0 / M
    t9 = dx * 1.0e3
    t7 = Il * t4
    t8 = R * dpsiA * r * t2
    t11 = l + l0 + t5
    t15 = R * dpsiA * r * t3 * 1.0e3
    t10 = Im + t7
    t12 = -t8
    t13 = t2 * t11
    t16 = dth * t3 * t11
    t14 = 1.0 / t10
    t17 = -t13
    t20 = dth * t13 * 1.0e3
    t21 = np.pi * (t13 - y) * -5.0
    t23 = dy + t12 + t16
    t18 = t17 + y
    t22 = np.tan(t21)
    t25 = t9 + t15 + t20
    t19 = np.abs(t18)
    t24 = t22 * 1.0e2
    t26 = np.arctan(t25)
    t27 = t19 * t23 * 5.0e3
    t28 = t24 + t27

    # Final output
    ddq_A = np.zeros(6)
    ddq_A[0] = t6 * (t26 * t28 * 0.5092958178940651 + M * g * np.sin(ramp))
    ddq_A[1] = -t6 * (t28 + M * g * np.cos(ramp))
    ddq_A[2] = -(t3 * t11 * t28 - t13 * t26 * t28 * 0.5092958178940651) / I
    ddq_A[3] = t14 * (uA + R * r * t2 * t28 + R * r * t3 * t26 * t28 * 0.5092958178940651)
    ddq_A[4] = t14 * uB
    ddq_A[5] = t14 * uF

    return ddq_A.reshape(6, 1)

def ddq_A_u(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to control inputs u for a rimless wheel system with spoke A
    in contact with the floor.
    """
    # Unpack inputs
    Il = in2[10]
    Im = in2[11]
    R = in2[9]

    # Intermediate calculations
    t2 = R ** 2
    t3 = Il * t2
    t4 = Im + t3
    t5 = 1.0 / t4

    # Output array with reshaping
    ddq_A_u = np.zeros((6, 3))
    ddq_A_u[3, 0] = t5
    ddq_A_u[4, 1] = t5
    ddq_A_u[5, 2] = t5

    return ddq_A_u

def ddq_A_q(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to q for a rimless wheel system with spoke A
    in contact with the floor.
    """
    # Unpack inputs
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    r = in2[8]
    th = in1[2]
    y = in1[1]

    # Intermediate calculations
    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R ** 2
    t5 = R * psiA * r
    t6 = 1.0 / I
    t7 = 1.0 / M
    t11 = dx * 1.0e+3
    t8 = Il * t4
    t9 = R * dpsiA * r * t2
    t10 = R * dpsiA * r * t3
    t13 = l + l0 + t5
    t12 = Im + t8
    t14 = -t9
    t15 = t2 * t13
    t16 = t9 * 1.0e+3
    t18 = t10 * 1.0e+3
    t21 = dth * t3 * t13
    t17 = 1.0 / t12
    t20 = dth * t15
    t22 = -t15
    t27 = t21 * 1.0e+3
    t28 = np.pi * (t15 - y) * -5.0
    t32 = dy + t14 + t21
    t23 = t22 + y
    t26 = t20 * 1.0e+3
    t29 = t10 + t20
    t30 = np.tan(t28)
    t24 = np.abs(t23)
    t25 = np.sign(t23)
    t31 = t30 ** 2
    t33 = t30 * 1.0e+2
    t40 = t11 + t18 + t26
    t34 = t31 + 1.0
    t35 = R * dth * r * t3 * t24 * 5.0e+3
    t41 = np.arctan(t40)
    t42 = t40 ** 2
    t45 = t24 * t29 * 5.0e+3
    t47 = t24 * t32 * 5.0e+3
    t48 = t25 * t32 * 5.0e+3
    t36 = -t35
    t38 = t34 * np.pi * 5.0e+2
    t44 = t42 + 1.0
    t49 = R * r * t2 * t48
    t50 = t3 * t13 * t48
    t51 = t33 + t47
    t39 = R * r * t2 * t38
    t43 = t3 * t13 * t38
    t46 = 1.0 / t44
    t52 = t38 + t48
    t53 = R * r * t3 * t51
    t57 = R * r * t2 * t41 * t51 * 5.092958178940651e-1
    t54 = -t53
    t55 = t36 + t39 + t49
    t56 = t43 + t45 + t50

    # Matrix computations
    mt1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t7 * t41 * t52 * 5.092958178940651e-1, -t7 * t52, -t6 * (t3 * t13 * t52 - t15 * t41 * t52 * 5.092958178940651e-1), t17 * (R * r * t2 * t52 + R * r * t3 * t41 * t52 * 5.092958178940651e-1), 0.0, 0.0])
    mt2 = np.array([t7 * (t41 * t56 * 5.092958178940651e-1 + t46 * t51 * (t16 - t27) * 5.092958178940651e-1), -t7 * t56])
    mt3 = np.array([-t6 * (t15 * t51 + t3 * t13 * t56 - t15 * t41 * t56 * 5.092958178940651e-1 + t3 * t13 * t41 * t51 * 5.092958178940651e-1 - t15 * t46 * t51 * (t16 - t27) * 5.092958178940651e-1)])
    mt4 = np.array([t17 * (t54 + t57 + t46 * t53 * (t16 - t27) * 5.092958178940651e-1 + R * r * t2 * t56 + R * r * t3 * t41 * t56 * 5.092958178940651e-1), 0.0, 0.0, -t7 * (t41 * t55 * 5.092958178940651e-1 - R * dth * r * t2 * t46 * t51 * 5.092958178940651e+2), t7 * t55])
    mt5 = np.array([t6 * (t54 + t57 + t3 * t13 * t55 - t15 * t41 * t55 * 5.092958178940651e-1 + R * r * t2 * t20 * t46 * t51 * 5.092958178940651e+2), -t17 * (R * r * t2 * t55 + R * r * t3 * t41 * t55 * 5.092958178940651e-1 - dth * r ** 2 * t2 * t3 * t4 * t46 * t51 * 5.092958178940651e+2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Reshaping and returning the result
    ddq_A_q = np.reshape(np.concatenate([mt1, mt2, mt3, mt4, mt5]), (6, 6))
    return ddq_A_q.T

def ddq_A_dq(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to dq for a rimless wheel system with spoke A
    in contact with the floor.
    """
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    r = in2[8]
    th = in1[2]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R**2
    t5 = r**2
    t8 = R * psiA * r
    t9 = 1.0 / I
    t10 = 1.0 / M
    t13 = dx * 1.0e+3
    t6 = t2**2
    t7 = t3**2
    t11 = Il * t4
    t12 = R * dpsiA * r * t2
    t15 = l + l0 + t8
    t20 = R * dpsiA * r * t3 * 1.0e+3
    t14 = Im + t11
    t16 = -t12
    t17 = t15**2
    t18 = t2 * t15
    t21 = dth * t3 * t15
    t19 = 1.0 / t14
    t22 = -t18
    t25 = dth * t18 * 1.0e+3
    t26 = np.pi * (t18 - y) * -5.0
    t28 = dy + t16 + t21
    t23 = t22 + y
    t27 = np.tan(t26)
    t30 = t13 + t20 + t25
    t24 = np.abs(t23)
    t29 = t27 * 1.0e+2
    t31 = np.arctan(t30)
    t33 = t30**2
    t32 = R * r * t3 * t18 * t24 * 5.0e+3
    t34 = t33 + 1.0
    t36 = t24 * t28 * 5.0e+3
    t35 = 1.0 / t34
    t37 = t29 + t36
    t38 = R * r * t3 * t18 * t35 * t37 * 5.092958178940651e+2

    mt1 = [t10 * t35 * t37 * 5.092958178940651e+2, 0.0, t9 * t18 * t35 * t37 * 5.092958178940651e+2, R * r * t3 * t19 * t35 * t37 * 5.092958178940651e+2, 0.0, 0.0, t10 * t24 * t31 * 2.546479089470326e+3, t10 * t24 * -5.0e+3]
    mt2 = [-t9 * (t3 * t15 * t24 * 5.0e+3 - t18 * t24 * t31 * 2.546479089470326e+3), t19 * (R * r * t2 * t24 * 5.0e+3 + R * r * t3 * t24 * t31 * 2.546479089470326e+3), 0.0, 0.0, t10 * (t18 * t35 * t37 * 5.092958178940651e+2 + t3 * t15 * t24 * t31 * 2.546479089470326e+3), t3 * t10 * t15 * t24 * -5.0e+3]
    mt3 = [t9 * (t7 * t17 * t24 * -5.0e+3 + t6 * t17 * t35 * t37 * 5.092958178940651e+2 + t2 * t3 * t17 * t24 * t31 * 2.546479089470326e+3), t19 * (t32 + t38 + R * r * t7 * t15 * t24 * t31 * 2.546479089470326e+3), 0.0, 0.0]
    mt4 = [-t10 * (R * r * t2 * t24 * t31 * 2.546479089470326e+3 - R * r * t3 * t35 * t37 * 5.092958178940651e+2), R * r * t2 * t10 * t24 * 5.0e+3, t9 * (t32 + t38 - R * r * t6 * t15 * t24 * t31 * 2.546479089470326e+3)]
    mt5 = [-t19 * (t4 * t5 * t6 * t24 * 5.0e+3 - t4 * t5 * t7 * t35 * t37 * 5.092958178940651e+2 + t2 * t3 * t4 * t5 * t24 * t31 * 2.546479089470326e+3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ddq_A_dq = np.reshape(np.concatenate([mt1, mt2, mt3, mt4, mt5]), (6, 6))
    return ddq_A_dq.T


def ddq_AB(in1, in2, in3):
    """
    Compute the base accelerations for a rimless wheel system with spokes A and B in contact with the floor.
    """
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dpsiB = in1[10]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    g = in2[4]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    psiB = in1[4]
    r = in2[8]
    ramp = in2[5]
    th = in1[2]
    uA = in3[0]
    uB = in3[1]
    uF = in3[2]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R ** 2
    t5 = R * psiA * r
    t6 = R * psiB * r
    t7 = 1.0 / M
    t9 = np.pi / 3.0
    t11 = dx * 1.0e+3
    t8 = Il * t4
    t10 = R * dpsiA * r * t2
    t13 = t9 + th
    t14 = l + l0 + t5
    t15 = l + l0 + t6
    t21 = R * dpsiA * r * t3 * 1.0e+3
    t12 = Im + t8
    t16 = -t10
    t17 = np.cos(t13)
    t18 = np.sin(t13)
    t19 = t2 * t14
    t22 = dth * t3 * t14
    t20 = 1.0 / t12
    t23 = -t19
    t24 = R * dpsiB * r * t17
    t28 = dth * t19 * 1.0e+3
    t29 = t15 * t17
    t30 = R * dpsiB * r * t18 * 1.0e+3
    t31 = dth * t15 * t18
    t32 = np.pi * (t19 - y) * -5.0
    t38 = dy + t16 + t22
    t25 = -t24
    t26 = t23 + y
    t33 = -t29
    t34 = np.tan(t32)
    t37 = dth * t29 * 1.0e+3
    t40 = np.pi * (t29 - y) * -5.0
    t43 = t11 + t21 + t28
    t27 = np.abs(t26)
    t35 = t33 + y
    t39 = t34 * 1.0e+2
    t41 = np.tan(t40)
    t44 = np.arctan(t43)
    t45 = dy + t25 + t31
    t47 = t11 + t30 + t37
    t36 = np.abs(t35)
    t42 = t41 * 1.0e+2
    t46 = t27 * t38 * 5.0e+3
    t48 = np.arctan(t47)
    t49 = t36 * t45 * 5.0e+3
    t50 = t39 + t46
    t51 = t42 + t49

    mt1 = [t7 * (t44 * t50 * 5.092958178940651e-1 + t48 * t51 * 5.092958178940651e-1 + M * g * np.sin(ramp)),
           -t7 * (t50 + t51 + M * g * np.cos(ramp)),
           -(
                       t3 * t14 * t50 + t15 * t18 * t51 - t19 * t44 * t50 * 5.092958178940651e-1 - t29 * t48 * t51 * 5.092958178940651e-1) / I]

    mt2 = [t20 * (uA + R * r * t2 * t50 + R * r * t3 * t44 * t50 * 5.092958178940651e-1),
           t20 * (uB + R * r * t17 * t51 + R * r * t18 * t48 * t51 * 5.092958178940651e-1),
           t20 * uF]

    ddq_AB = np.hstack([mt1, mt2])
    return ddq_AB.reshape(6, 1)


def ddq_AB_u(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to control inputs u for a rimless wheel system with spokes A
    and B in contact with the floor.
    """
    Il = in2[10]
    Im = in2[11]
    R = in2[9]
    t2 = R**2
    t3 = Il * t2
    t4 = Im + t3
    t5 = 1.0 / t4
    ddq_AB_u = np.zeros((6, 3))
    ddq_AB_u[3, 0] = t5
    ddq_AB_u[4, 1] = t5
    ddq_AB_u[5, 2] = t5
    return ddq_AB_u


def ddq_AB_q(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to q for a rimless wheel system with spokes A
    and B in contact with the floor.
    """
    # Extract inputs
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    r = in2[8]
    l = in2[3]
    l0 = in2[7]

    # Additional inputs
    dx = in1[6]
    dy = in1[7]
    dth = in1[8]
    dpsiA = in1[9]
    dpsiB = in1[10]
    th = in1[2]
    psiA = in1[3]
    psiB = in1[4]
    y = in1[1]

    # Compute intermediate terms
    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R ** 2
    t5 = r ** 2
    t6 = R * psiA * r
    t7 = R * psiB * r
    t8 = 1 / I
    t9 = 1 / M
    t11 = np.pi / 3.0
    t14 = dx * 1.0e+3
    t10 = Il * t4
    t12 = R * dpsiA * r * t2
    t13 = R * dpsiA * r * t3
    t16 = t11 + th
    t17 = l + l0 + t6
    t18 = l + l0 + t7
    t15 = Im + t10
    t19 = -t12
    t20 = np.cos(t16)
    t21 = np.sin(t16)
    t22 = t2 * t17
    t23 = t12 * 1.0e+3
    t25 = t13 * 1.0e+3
    t28 = dth * t3 * t17
    t24 = 1.0 / t15
    t27 = dth * t22
    t29 = -t22
    t30 = R * dpsiB * r * t20
    t31 = R * dpsiB * r * t21
    t37 = t28 * 1.0e+3
    t38 = t18 * t20
    t43 = dth * t18 * t21
    t44 = np.pi * (t22 - y) * -5.0
    t54 = dy + t19 + t28
    t32 = -t30
    t33 = t29 + y
    t36 = t27 * 1.0e+3
    t39 = t30 * 1.0e+3
    t40 = t31 * 1.0e+3
    t42 = dth * t38
    t45 = -t38
    t46 = t13 + t27
    t47 = np.tan(t44)
    t53 = t43 * 1.0e+3
    t57 = np.pi * (t38 - y) * -5.0
    t34 = np.abs(t33)
    t35 = np.sign(t33)
    t48 = t47 ** 2
    t49 = t45 + y
    t52 = t42 * 1.0e+3
    t55 = t47 * 1.0e+2
    t58 = np.tan(t57)
    t66 = t31 + t42
    t68 = t14 + t25 + t36
    t71 = dy + t32 + t43
    t50 = np.abs(t49)
    t51 = np.sign(t49)
    t56 = t48 + 1.0
    t59 = R * dth * r * t3 * t34 * 5.0e+3
    t60 = t58 ** 2
    t64 = t58 * 1.0e+2
    t69 = np.arctan(t68)
    t72 = t68 ** 2
    t77 = t34 * t46 * 5.0e+3
    t80 = t34 * t54 * 5.0e+3
    t81 = t35 * t54 * 5.0e+3
    t83 = t14 + t40 + t52
    t61 = -t59
    t63 = t56 * np.pi * 5.0e+2
    t65 = t60 + 1.0
    t74 = t72 + 1.0
    t75 = R * dth * r * t21 * t50 * 5.0e+3
    t84 = np.arctan(t83)
    t85 = R * r * t2 * t81
    t86 = t83 ** 2
    t89 = t3 * t17 * t81
    t91 = t50 * t66 * 5.0e+3
    t92 = t50 * t71 * 5.0e+3
    t93 = t51 * t71 * 5.0e+3
    t94 = t55 + t80
    t67 = R * r * t2 * t63
    t70 = t65 * np.pi * 5.0e+2
    t73 = t3 * t17 * t63
    t76 = -t75
    t78 = 1.0 / t74
    t88 = t86 + 1.0
    t95 = t63 + t81
    t96 = R * r * t20 * t93
    t97 = R * r * t3 * t94
    t99 = t18 * t21 * t93
    t100 = t64 + t92
    t106 = R * r * t2 * t69 * t94 * 5.092958178940651e-1
    t82 = R * r * t20 * t70
    t87 = t18 * t21 * t70
    t90 = 1.0 / t88
    t98 = -t97
    t101 = t70 + t93
    t102 = R * r * t21 * t100
    t104 = t61 + t67 + t85
    t105 = t73 + t77 + t89
    t108 = R * r * t20 * t84 * t100 * 5.092958178940651e-1
    t103 = -t102
    t107 = t76 + t82 + t96
    t109 = t87 + t91 + t99

    et1 = t22 * t94 + t38 * t100 + t3 * t17 * t105 + t18 * t21 * t109 - t22 * t69 * t105 * 5.092958178940651e-1 - t38 * t84 * t109 * 5.092958178940651e-1 + t3 * t17 * t69 * t94 * 5.092958178940651e-1 + t18 * t21 * t84 * t100 * 5.092958178940651e-1
    et2 = t22 * t78 * t94 * (t23 - t37) * -5.092958178940651e-1 - t38 * t90 * t100 * (t39 - t53) * 5.092958178940651e-1

    mt1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t9 * (t69 * t95 * 5.092958178940651e-1 + t84 * t101 * 5.092958178940651e-1),
           -t9 * (t95 + t101), -t8 * (
                       t3 * t17 * t95 + t18 * t21 * t101 - t22 * t69 * t95 * 5.092958178940651e-1 - t38 * t84 * t101 * 5.092958178940651e-1)]
    mt2 = [t24 * (R * r * t2 * t95 + R * r * t3 * t69 * t95 * 5.092958178940651e-1),
           t24 * (R * r * t20 * t101 + R * r * t21 * t84 * t101 * 5.092958178940651e-1), 0.0]
    mt3 = [t9 * (t69 * t105 * 5.092958178940651e-1 + t84 * t109 * 5.092958178940651e-1 + t78 * t94 * (
                t23 - t37) * 5.092958178940651e-1 + t90 * t100 * (t39 - t53) * 5.092958178940651e-1),
           -t9 * (t105 + t109), -t8 * (et1 + et2)]
    mt4 = [t24 * (t98 + t106 + t78 * t97 * (
                t23 - t37) * 5.092958178940651e-1 + R * r * t2 * t105 + R * r * t3 * t69 * t105 * 5.092958178940651e-1),
           t24 * (t103 + t108 + t90 * t102 * (
                       t39 - t53) * 5.092958178940651e-1 + R * r * t20 * t109 + R * r * t21 * t84 * t109 * 5.092958178940651e-1),
           0.0]
    mt5 = [-t9 * (t69 * t104 * 5.092958178940651e-1 - R * dth * r * t2 * t78 * t94 * 5.092958178940651e+2), t9 * t104,
           t8 * (
                       t98 + t106 + t3 * t17 * t104 - t22 * t69 * t104 * 5.092958178940651e-1 + R * r * t2 * t27 * t78 * t94 * 5.092958178940651e+2)]
    mt6 = [-t24 * (
                R * r * t2 * t104 + R * r * t3 * t69 * t104 * 5.092958178940651e-1 - dth * t2 * t3 * t4 * t5 * t78 * t94 * 5.092958178940651e+2),
           0.0, 0.0, -t9 * (t84 * t107 * 5.092958178940651e-1 - R * dth * r * t20 * t90 * t100 * 5.092958178940651e+2),
           t9 * t107]
    mt7 = [t8 * (
                t103 + t108 + t18 * t21 * t107 - t38 * t84 * t107 * 5.092958178940651e-1 + R * r * t20 * t42 * t90 * t100 * 5.092958178940651e+2),
           0.0, -t24 * (
                       R * r * t20 * t107 + R * r * t21 * t84 * t107 * 5.092958178940651e-1 - dth * t4 * t5 * t20 * t21 * t90 * t100 * 5.092958178940651e+2),
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ddq_AB_q = np.array(np.concatenate([mt1, mt2, mt3, mt4, mt5, mt6, mt7])).reshape(6, 6)
    return ddq_AB_q.T


def ddq_AB_dq(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to dq for a rimless wheel system with spokes A
    and B in contact with the floor.
    """
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dpsiB = in1[10]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    psiB = in1[4]
    r = in2[8]
    th = in1[2]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R ** 2
    t5 = r ** 2
    t8 = R * psiA * r
    t9 = R * psiB * r
    t10 = 1.0 / I
    t11 = 1.0 / M
    t13 = np.pi / 3.0
    t15 = dx * 1.0e+3
    t6 = t2 ** 2
    t7 = t3 ** 2
    t12 = Il * t4
    t14 = R * dpsiA * r * t2
    t17 = t13 + th
    t18 = l + l0 + t8
    t19 = l + l0 + t9
    t29 = R * dpsiA * r * t3 * 1.0e+3
    t16 = Im + t12
    t20 = -t14
    t21 = np.cos(t17)
    t22 = np.sin(t17)
    t23 = t18 ** 2
    t24 = t19 ** 2
    t25 = t2 * t18
    t30 = dth * t3 * t18
    t26 = t21 ** 2
    t27 = t22 ** 2
    t28 = 1.0 / t16
    t31 = -t25
    t32 = R * dpsiB * r * t21
    t36 = dth * t25 * 1.0e+3
    t37 = t19 * t21
    t38 = R * dpsiB * r * t22 * 1.0e+3
    t39 = dth * t19 * t22
    t40 = np.pi * (t25 - y) * -5.0
    t46 = dy + t20 + t30
    t33 = -t32
    t34 = t31 + y
    t41 = -t37
    t42 = np.tan(t40)
    t45 = dth * t37 * 1.0e+3
    t48 = np.pi * (t37 - y) * -5.0
    t52 = t15 + t29 + t36
    t35 = np.abs(t34)
    t43 = t41 + y
    t47 = t42 * 1.0e+2
    t49 = np.tan(t48)
    t53 = np.arctan(t52)
    t54 = dy + t33 + t39
    t56 = t52 ** 2
    t61 = t15 + t38 + t45
    t44 = np.abs(t43)
    t50 = t49 * 1.0e+2
    t51 = t3 * t18 * t35 * 5.0e+3
    t55 = R * r * t3 * t25 * t35 * 5.0e+3
    t57 = t56 + 1.0
    t59 = t35 * t46 * 5.0e+3
    t62 = np.arctan(t61)
    t63 = t61 ** 2
    t58 = 1.0 / t57
    t60 = t19 * t22 * t44 * 5.0e+3
    t64 = t63 + 1.0
    t65 = R * r * t22 * t37 * t44 * 5.0e+3
    t67 = t44 * t54 * 5.0e+3
    t68 = t47 + t59
    t66 = 1.0 / t64
    t69 = t50 + t67
    t70 = t25 * t58 * t68 * 5.092958178940651e+2
    t71 = R * r * t3 * t70
    t72 = t37 * t66 * t69 * 5.092958178940651e+2
    t73 = R * r * t22 * t72

    mt1 = [t11 * (t58 * t68 * 5.092958178940651e+2 + t66 * t69 * 5.092958178940651e+2), 0.0, t10 * (t70 + t72),
           R * r * t3 * t28 * t58 * t68 * 5.092958178940651e+2, R * r * t22 * t28 * t66 * t69 * 5.092958178940651e+2,
           0.0]
    mt2 = [t11 * (t35 * t53 * 2.546479089470326e+3 + t44 * t62 * 2.546479089470326e+3),
           -t11 * (t35 * 5.0e+3 + t44 * 5.0e+3),
           -t10 * (t51 + t60 - t25 * t35 * t53 * 2.546479089470326e+3 - t37 * t44 * t62 * 2.546479089470326e+3)]
    mt3 = [t28 * (R * r * t2 * t35 * 5.0e+3 + R * r * t3 * t35 * t53 * 2.546479089470326e+3),
           t28 * (R * r * t21 * t44 * 5.0e+3 + R * r * t22 * t44 * t62 * 2.546479089470326e+3), 0.0, t11 * (
                       t70 + t72 + t3 * t18 * t35 * t53 * 2.546479089470326e+3 + t19 * t22 * t44 * t62 * 2.546479089470326e+3),
           -t11 * (t51 + t60)]
    mt4 = [t10 * (
                t7 * t23 * t35 * -5.0e+3 - t24 * t27 * t44 * 5.0e+3 + t6 * t23 * t58 * t68 * 5.092958178940651e+2 + t24 * t26 * t66 * t69 * 5.092958178940651e+2 + t2 * t3 * t23 * t35 * t53 * 2.546479089470326e+3 + t21 * t22 * t24 * t44 * t62 * 2.546479089470326e+3)]
    mt5 = [t28 * (t55 + t71 + R * r * t7 * t18 * t35 * t53 * 2.546479089470326e+3),
           t28 * (t65 + t73 + R * r * t19 * t27 * t44 * t62 * 2.546479089470326e+3), 0.0,
           -t11 * (R * r * t2 * t35 * t53 * 2.546479089470326e+3 - R * r * t3 * t58 * t68 * 5.092958178940651e+2),
           R * r * t2 * t11 * t35 * 5.0e+3]
    mt6 = [t10 * (t55 + t71 - R * r * t6 * t18 * t35 * t53 * 2.546479089470326e+3), -t28 * (
                t4 * t5 * t6 * t35 * 5.0e+3 - t4 * t5 * t7 * t58 * t68 * 5.092958178940651e+2 + t2 * t3 * t4 * t5 * t35 * t53 * 2.546479089470326e+3),
           0.0, 0.0]
    mt7 = [-t11 * (R * r * t21 * t44 * t62 * 2.546479089470326e+3 - R * r * t22 * t66 * t69 * 5.092958178940651e+2),
           R * r * t11 * t21 * t44 * 5.0e+3, t10 * (t65 + t73 - R * r * t19 * t26 * t44 * t62 * 2.546479089470326e+3),
           0.0]
    mt8 = [-t28 * (
                t4 * t5 * t26 * t44 * 5.0e+3 - t4 * t5 * t27 * t66 * t69 * 5.092958178940651e+2 + t4 * t5 * t21 * t22 * t44 * t62 * 2.546479089470326e+3),
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ddq_AB_dq = np.reshape(np.concatenate([mt1, mt2, mt3, mt4, mt5, mt6, mt7, mt8]), (6, 6))

    return ddq_AB_dq.T

def ddq_AF(in1, in2, in3):
    """
    Compute the base accelerations for a rimless wheel system with spokes A and F in contact with the floor.
    """
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dpsiF = in1[11]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    g = in2[4]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    psiF = in1[5]
    r = in2[8]
    ramp = in2[5]
    th = in1[2]
    uA, uB, uF = in3
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R**2
    t5 = R * psiA * r
    t6 = R * psiF * r
    t7 = 1.0 / M
    t9 = np.pi * (5.0 / 3.0)
    t11 = dx * 1.0e3
    t8 = Il * t4
    t10 = R * dpsiA * r * t2
    t13 = t9 + th
    t14 = l + l0 + t5
    t15 = l + l0 + t6
    t21 = R * dpsiA * r * t3 * 1.0e3
    t12 = Im + t8
    t16 = -t10
    t17 = np.cos(t13)
    t18 = np.sin(t13)
    t19 = t2 * t14
    t22 = dth * t3 * t14
    t20 = 1.0 / t12
    t23 = -t19
    t24 = R * dpsiF * r * t17
    t28 = dth * t19 * 1.0e3
    t29 = t15 * t17
    t30 = R * dpsiF * r * t18 * 1.0e3
    t31 = dth * t15 * t18
    t32 = np.pi * (t19 - y) * -5.0
    t38 = dy + t16 + t22
    t25 = -t24
    t26 = t23 + y
    t33 = -t29
    t34 = np.tan(t32)
    t37 = dth * t29 * 1.0e3
    t40 = np.pi * (t29 - y) * -5.0
    t43 = t11 + t21 + t28
    t27 = np.abs(t26)
    t35 = t33 + y
    t39 = t34 * 1.0e2
    t41 = np.tan(t40)
    t44 = np.arctan(t43)
    t45 = dy + t25 + t31
    t47 = t11 + t30 + t37
    t36 = np.abs(t35)
    t42 = t41 * 1.0e2
    t46 = t27 * t38 * 5.0e3
    t48 = np.arctan(t47)
    t49 = t36 * t45 * 5.0e3
    t50 = t39 + t46
    t51 = t42 + t49

    mt1_1 = t7 * (t44 * t50 * 0.5092958178940651 + t48 * t51 * 0.5092958178940651 + M * g * np.sin(ramp))
    mt1_2 = -t7 * (t50 + t51 + M * g * np.cos(ramp))
    mt1_3 = -(t3 * t14 * t50 + t15 * t18 * t51 - t19 * t44 * t50 * 0.5092958178940651 - t29 * t48 * t51 * 0.5092958178940651) / I

    mt2_1 = t20 * (uA + R * r * t2 * t50 + R * r * t3 * t44 * t50 * 0.5092958178940651)
    mt2_2 = t20 * uB
    mt2_3 = t20 * (uF + R * r * t17 * t51 + R * r * t18 * t48 * t51 * 0.5092958178940651)

    ddq_AF = np.array([mt1_1, mt1_2, mt1_3, mt2_1, mt2_2, mt2_3]).reshape(6, 1)
    return ddq_AF


def ddq_AF_u(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to control inputs u for a rimless wheel system with spokes A
    and F in contact with the floor.
    """
    R = in2[9]
    Il = in2[10]
    Im = in2[11]

    t2 = R ** 2
    t3 = Il * t2
    t4 = Im + t3
    t5 = 1.0 / t4

    ddq_AF_u = np.zeros((6, 3))
    ddq_AF_u[3, 0] = t5
    ddq_AF_u[4, 1] = t5
    ddq_AF_u[5, 2] = t5

    return ddq_AF_u


def ddq_AF_q(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to q for a rimless wheel system with spokes A
    and F in contact with the floor.
    """
    # Unpack parameters
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dpsiF = in1[11]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    psiF = in1[5]
    r = in2[8]
    th = in1[2]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R ** 2
    t5 = r ** 2
    t6 = R * psiA * r
    t7 = R * psiF * r
    t8 = 1.0 / I
    t9 = 1.0 / M
    t11 = np.pi * (5.0 / 3.0)
    t14 = dx * 1.0e+3
    t10 = Il * t4
    t12 = R * dpsiA * r * t2
    t13 = R * dpsiA * r * t3
    t16 = t11 + th
    t17 = l + l0 + t6
    t18 = l + l0 + t7
    t15 = Im + t10
    t19 = -t12
    t20 = np.cos(t16)
    t21 = np.sin(t16)
    t22 = t2 * t17
    t23 = t12 * 1.0e+3
    t25 = t13 * 1.0e+3
    t28 = dth * t3 * t17
    t24 = 1.0 / t15
    t27 = dth * t22
    t29 = -t22
    t30 = R * dpsiF * r * t20
    t31 = R * dpsiF * r * t21
    t37 = t28 * 1.0e+3
    t38 = t18 * t20
    t43 = dth * t18 * t21
    t44 = np.pi * (t22 - y) * -5.0
    t54 = dy + t19 + t28
    t32 = -t30
    t33 = t29 + y
    t36 = t27 * 1.0e+3
    t39 = t30 * 1.0e+3
    t40 = t31 * 1.0e+3
    t42 = dth * t38
    t45 = -t38
    t46 = t13 + t27
    t47 = np.tan(t44)
    t53 = t43 * 1.0e+3
    t57 = np.pi * (t38 - y) * -5.0
    t34 = np.abs(t33)
    t35 = np.sign(t33)
    t48 = t47 ** 2
    t49 = t45 + y
    t52 = t42 * 1.0e+3
    t55 = t47 * 1.0e+2
    t58 = np.tan(t57)
    t66 = t31 + t42
    t68 = t14 + t25 + t36
    t71 = dy + t32 + t43
    t50 = np.abs(t49)
    t51 = np.sign(t49)
    t56 = t48 + 1.0
    t59 = R * dth * r * t3 * t34 * 5.0e+3
    t60 = t58 ** 2
    t64 = t58 * 1.0e+2
    t69 = np.arctan(t68)
    t72 = t68 ** 2
    t77 = t34 * t46 * 5.0e+3
    t80 = t34 * t54 * 5.0e+3
    t81 = t35 * t54 * 5.0e+3
    t83 = t14 + t40 + t52
    t61 = -t59
    t63 = t56 * np.pi * 5.0e+2
    t65 = t60 + 1.0
    t74 = t72 + 1.0
    t75 = R * dth * r * t21 * t50 * 5.0e+3
    t84 = np.arctan(t83)
    t85 = R * r * t2 * t81
    t86 = t83 ** 2
    t89 = t3 * t17 * t81
    t91 = t50 * t66 * 5.0e+3
    t92 = t50 * t71 * 5.0e+3
    t93 = t51 * t71 * 5.0e+3
    t94 = t55 + t80
    t67 = R * r * t2 * t63
    t70 = t65 * np.pi * 5.0e+2
    t73 = t3 * t17 * t63
    t76 = -t75
    t78 = 1.0 / t74
    t88 = t86 + 1.0
    t95 = t63 + t81
    t96 = R * r * t20 * t93
    t97 = R * r * t3 * t94
    t99 = t18 * t21 * t93
    t100 = t64 + t92
    t106 = R * r * t2 * t69 * t94 * 5.092958178940651e-1
    t82 = R * r * t20 * t70
    t87 = t18 * t21 * t70
    t90 = 1.0 / t88
    t98 = -t97
    t101 = t70 + t93
    t102 = R * r * t21 * t100
    t104 = t61 + t67 + t85
    t105 = t73 + t77 + t89
    t108 = R * r * t20 * t84 * t100 * 5.092958178940651e-1
    t103 = -t102
    t107 = t76 + t82 + t96
    t109 = t87 + t91 + t99

    et1 = t22 * t94 + t38 * t100 + t3 * t17 * t105 + t18 * t21 * t109 - t22 * t69 * t105 * 5.092958178940651e-1 - t38 * t84 * t109 * 5.092958178940651e-1 + t3 * t17 * t69 * t94 * 5.092958178940651e-1 + t18 * t21 * t84 * t100 * 5.092958178940651e-1
    et2 = t22 * t78 * t94 * (t23 - t37) * -5.092958178940651e-1 - t38 * t90 * t100 * (t39 - t53) * 5.092958178940651e-1

    mt1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t9 * (t69 * t95 * 5.092958178940651e-1 + t84 * t101 * 5.092958178940651e-1),
         -t9 * (t95 + t101), -t8 * (
                     t3 * t17 * t95 + t18 * t21 * t101 - t22 * t69 * t95 * 5.092958178940651e-1 - t38 * t84 * t101 * 5.092958178940651e-1)])
    mt2 = np.array([t24 * (R * r * t2 * t95 + R * r * t3 * t69 * t95 * 5.092958178940651e-1), 0.0,
                    t24 * (R * r * t20 * t101 + R * r * t21 * t84 * t101 * 5.092958178940651e-1)])
    mt3 = np.array([t9 * (t69 * t105 * 5.092958178940651e-1 + t84 * t109 * 5.092958178940651e-1 + t78 * t94 * (
                t23 - t37) * 5.092958178940651e-1 + t90 * t100 * (t39 - t53) * 5.092958178940651e-1),
                    -t9 * (t105 + t109), -t8 * (et1 + et2)])
    mt4 = np.array([t24 * (t98 + t106 + t78 * t97 * (
                t23 - t37) * 5.092958178940651e-1 + R * r * t2 * t105 + R * r * t3 * t69 * t105 * 5.092958178940651e-1),
                    0.0, t24 * (t103 + t108 + t90 * t102 * (
                    t39 - t53) * 5.092958178940651e-1 + R * r * t20 * t109 + R * r * t21 * t84 * t109 * 5.092958178940651e-1)])
    mt5 = np.array(
        [-t9 * (t69 * t104 * 5.092958178940651e-1 - R * dth * r * t2 * t78 * t94 * 5.092958178940651e+2), t9 * t104,
         t8 * (
                     t98 + t106 + t3 * t17 * t104 - t22 * t69 * t104 * 5.092958178940651e-1 + R * r * t2 * t27 * t78 * t94 * 5.092958178940651e+2)])
    mt6 = np.array([-t24 * (
                R * r * t2 * t104 + R * r * t3 * t69 * t104 * 5.092958178940651e-1 - dth * t2 * t3 * t4 * t5 * t78 * t94 * 5.092958178940651e+2),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    -t9 * (t84 * t107 * 5.092958178940651e-1 - R * dth * r * t20 * t90 * t100 * 5.092958178940651e+2),
                    t9 * t107])
    mt7 = np.array([t8 * (
                t103 + t108 + t18 * t21 * t107 - t38 * t84 * t107 * 5.092958178940651e-1 + R * r * t20 * t42 * t90 * t100 * 5.092958178940651e+2),
                    0.0, 0.0, -t24 * (
                                R * r * t20 * t107 + R * r * t21 * t84 * t107 * 5.092958178940651e-1 - dth * t4 * t5 * t20 * t21 * t90 * t100 * 5.092958178940651e+2)])

    ddq_AF_q = np.hstack([mt1, mt2, mt3, mt4, mt5, mt6, mt7]).reshape((6,6), order='F')
    return ddq_AF_q


def ddq_AF_dq(in1, in2):
    """
    Compute the Jacobian of base accelerations with respect to dq for a rimless wheel system with spokes A
    and F in contact with the floor.
    """
    I = in2[2]
    Il = in2[10]
    Im = in2[11]
    M = in2[0]
    R = in2[9]
    dpsiA = in1[9]
    dpsiF = in1[11]
    dth = in1[8]
    dx = in1[6]
    dy = in1[7]
    l = in2[3]
    l0 = in2[7]
    psiA = in1[3]
    psiF = in1[5]
    r = in2[8]
    th = in1[2]
    y = in1[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = R ** 2
    t5 = r ** 2
    t8 = R * psiA * r
    t9 = R * psiF * r
    t10 = 1.0 / I
    t11 = 1.0 / M
    t13 = np.pi * (5.0 / 3.0)
    t15 = dx * 1.0e3
    t6 = t2 ** 2
    t7 = t3 ** 2
    t12 = Il * t4
    t14 = R * dpsiA * r * t2
    t17 = t13 + th
    t18 = l + l0 + t8
    t19 = l + l0 + t9
    t29 = R * dpsiA * r * t3 * 1.0e3
    t16 = Im + t12
    t20 = -t14
    t21 = np.cos(t17)
    t22 = np.sin(t17)
    t23 = t18 ** 2
    t24 = t19 ** 2
    t25 = t2 * t18
    t30 = dth * t3 * t18
    t26 = t21 ** 2
    t27 = t22 ** 2
    t28 = 1.0 / t16
    t31 = -t25
    t32 = R * dpsiF * r * t21
    t36 = dth * t25 * 1.0e3
    t37 = t19 * t21
    t38 = R * dpsiF * r * t22 * 1.0e3
    t39 = dth * t19 * t22
    t40 = np.pi * (t25 - y) * -5.0
    t46 = dy + t20 + t30
    t33 = -t32
    t34 = t31 + y
    t41 = -t37
    t42 = np.tan(t40)
    t45 = dth * t37 * 1.0e3
    t48 = np.pi * (t37 - y) * -5.0
    t52 = t15 + t29 + t36
    t35 = np.abs(t34)
    t43 = t41 + y
    t47 = t42 * 1.0e2
    t49 = np.tan(t48)
    t53 = np.arctan(t52)
    t54 = dy + t33 + t39
    t56 = t52 ** 2
    t61 = t15 + t38 + t45
    t44 = np.abs(t43)
    t50 = t49 * 1.0e2
    t51 = t3 * t18 * t35 * 5.0e3
    t55 = R * r * t3 * t25 * t35 * 5.0e3
    t57 = t56 + 1.0
    t59 = t35 * t46 * 5.0e3
    t62 = np.arctan(t61)
    t63 = t61 ** 2
    t58 = 1.0 / t57
    t60 = t19 * t22 * t44 * 5.0e3
    t64 = t63 + 1.0
    t65 = R * r * t22 * t37 * t44 * 5.0e3
    t67 = t44 * t54 * 5.0e3
    t68 = t47 + t59
    t66 = 1.0 / t64
    t69 = t50 + t67
    t70 = t25 * t58 * t68 * 5.092958178940651e2
    t71 = R * r * t3 * t70
    t72 = t37 * t66 * t69 * 5.092958178940651e2
    t73 = R * r * t22 * t72

    mt1 = [t11 * (t58 * t68 * 5.092958178940651e2 + t66 * t69 * 5.092958178940651e2), 0.0, t10 * (t70 + t72),
           R * r * t3 * t28 * t58 * t68 * 5.092958178940651e2, 0.0, R * r * t22 * t28 * t66 * t69 * 5.092958178940651e2]
    mt2 = [t11 * (t35 * t53 * 2.546479089470326e3 + t44 * t62 * 2.546479089470326e3),
           -t11 * (t35 * 5.0e3 + t44 * 5.0e3),
           -t10 * (t51 + t60 - t25 * t35 * t53 * 2.546479089470326e3 - t37 * t44 * t62 * 2.546479089470326e3)]
    mt3 = [t28 * (R * r * t2 * t35 * 5.0e3 + R * r * t3 * t35 * t53 * 2.546479089470326e3), 0.0,
           t28 * (R * r * t21 * t44 * 5.0e3 + R * r * t22 * t44 * t62 * 2.546479089470326e3),
           t11 * (t70 + t72 + t3 * t18 * t35 * t53 * 2.546479089470326e3 + t19 * t22 * t44 * t62 * 2.546479089470326e3),
           -t11 * (t51 + t60)]
    mt4 = [t10 * (
                t7 * t23 * t35 * -5.0e3 - t24 * t27 * t44 * 5.0e3 + t6 * t23 * t58 * t68 * 5.092958178940651e2 + t24 * t26 * t66 * t69 * 5.092958178940651e2 + t2 * t3 * t23 * t35 * t53 * 2.546479089470326e3 + t21 * t22 * t24 * t44 * t62 * 2.546479089470326e3)]
    mt5 = [t28 * (t55 + t71 + R * r * t7 * t18 * t35 * t53 * 2.546479089470326e3), 0.0,
           t28 * (t65 + t73 + R * r * t19 * t27 * t44 * t62 * 2.546479089470326e3),
           -t11 * (R * r * t2 * t35 * t53 * 2.546479089470326e3 - R * r * t3 * t58 * t68 * 5.092958178940651e2),
           R * r * t2 * t11 * t35 * 5.0e3]
    mt6 = [t10 * (t55 + t71 - R * r * t6 * t18 * t35 * t53 * 2.546479089470326e3), -t28 * (
                t4 * t5 * t6 * t35 * 5.0e3 - t4 * t5 * t7 * t58 * t68 * 5.092958178940651e2 + t2 * t3 * t4 * t5 * t35 * t53 * 2.546479089470326e3),
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mt7 = [-t11 * (R * r * t21 * t44 * t62 * 2.546479089470326e3 - R * r * t22 * t66 * t69 * 5.092958178940651e2),
           R * r * t11 * t21 * t44 * 5.0e3, t10 * (t65 + t73 - R * r * t19 * t26 * t44 * t62 * 2.546479089470326e3),
           0.0, 0.0]
    mt8 = [-t28 * (
                t4 * t5 * t26 * t44 * 5.0e3 - t4 * t5 * t27 * t66 * t69 * 5.092958178940651e2 + t4 * t5 * t21 * t22 * t44 * t62 * 2.546479089470326e3)]

    ddq_AF_dq = np.hstack([mt1, mt2, mt3, mt4, mt5, mt6, mt7, mt8]).reshape((6, 6), order='F')

    return ddq_AF_dq

