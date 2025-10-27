import numpy as np
from simulation_helper_functions import *
from linearized_functions import *
import matplotlib.pyplot as plt
from gurobi_functions import *
from scipy import sparse

plt.rcParams['text.usetex'] = True
import time

def dynamics(z, p, yC, u_MPC):
    # Constants
    K_z = 1.0e2
    D_z = 5.0e3
    l_max = 0.1
    fr_coeff = 0.8

    # Get mass matrix
    A = A_rimless_wheel(z, p)

    # Get forces
    b = b_rimless_wheel(z, p)

    # Get contact forces
    forces = contact_force(z, p, yC, K_z, D_z, l_max, fr_coeff)

    # Compute generalized forces
    QF_A = jacobian_A(z, p).T @ forces[:, 0]
    QF_B = jacobian_B(z, p).T @ forces[:, 1]
    QF_C = jacobian_C(z, p).T @ forces[:, 2]
    QF_D = jacobian_D(z, p).T @ forces[:, 3]
    QF_E = jacobian_E(z, p).T @ forces[:, 4]
    QF_F = jacobian_F(z, p).T @ forces[:, 5]

    Q = QF_A + QF_B + QF_C + QF_D + QF_E + QF_F

    B = np.zeros((9,6))
    B[3, 0] = 1
    B[4, 1] = 1
    B[5, 2] = 1
    B[6, 3] = 1
    B[7, 4] = 1
    B[8, 5] = 1

    # only used for spokes not controlled by u_MPC
    u = control_law_passive(z, b, Q, u_MPC, u_params)

    # Solve for qdd
    B_u = B @ u
    qdd = np.linalg.pinv(A) @ (b + Q + B_u.reshape(9,))

    # Form dz
    dz = np.zeros_like(z)
    dz[:9] = z[9:]
    dz[9:] = qdd

    return dz, forces, u


def contact_force(z, p, yC, K_z, D_z, l_max, fr_coeff):
    keypoints = keypoints_rimless_wheel(z, p)
    keypoints_vel = keypoints_vel_rimless_wheel(z, p)
    forces = np.zeros((2, 6))

    for i in range(np.shape(keypoints)[1]):
        rN = keypoints[:, i]
        C_l = rN[1] - yC

        if C_l < 0:
            dC_l = keypoints_vel[:, i]

            F_N = -K_z * np.tan(np.pi * C_l / (2 * l_max)) - D_z * np.abs(C_l) * dC_l[1]
            F_f = -(2 / np.pi) * np.arctan(dC_l[0] / 0.001) * fr_coeff * F_N

            forces[:, i] = [F_f, F_N]

    return forces


def control_law_MPC(z, p, t_horizon, koop_model, norm_vals, objective_parameters):

    theta = z[2]
    theta_dot = z[11]

    x0 = np.zeros([12, ]) #[x y, theta, psiA, psiB, psiF, xdot, ydot, thetadot, psiAdot, psiBdot, psiFdot]
    x0[1] = z[1]  # copy over y-coord
    x0[2] = np.mod(theta, np.pi / 3)  # cast theta into 0<= theta <= pi/3
    x0[6:9] = z[9:12]  # copy over xdot, ydot, thetadot

    z_tie_breaker = z.copy()
    z_tie_breaker[0] = 0

    keypoints = keypoints_spokes_rimless_wheel(z_tie_breaker, p)
    indices = np.argsort(keypoints[1, :])[:3]

    x0[3] = z[3 + indices[0]]  # pseudo psi_A
    x0[3+6] = z[12 + indices[0]]  # pseudo psi_A_dot

    if indices[0] == 5:
        x0[4] = z[3 + 0]  # pseudo psi_B
        x0[5] = z[3 + 4]  # pseudo psi_F
        x0[4 + 6] = z[12 + 0]  # pseudo psi_B_dot
        x0[5 + 6] = z[12 + 4]  # pseudo psi_F_dot
    elif indices[0] == 0:
        x0[4] = z[3 + 1]  # pseudo psi_B
        x0[5] = z[3 + 5]  # pseudo psi_F
        x0[4 + 6] = z[12 + 1]  # pseudo psi_B_dot
        x0[5 + 6] = z[12 + 5]  # pseudo psi_F_dot
    else:
        x0[4] = z[3 + np.max(indices[1:3])]  # pseudo psi_B
        x0[5] = z[3 + np.min(indices[1:3])]  # pseudo psi_F
        x0[4 + 6] = z[12 + np.max(indices[1:3])]  # pseudo psi_B_dot
        x0[5 + 6] = z[12 + np.min(indices[1:3])]  # pseudo psi_F_dot

    if keypoints[0, indices[0]] < 0:
        x0[2] = x0[2] - np.pi/3

    # finally convert x0 to normalized coordinates
    x0_normalized = x0/norm_vals

    control, mpc_x_output, traj_normalized, obj, wse = mpc_cck(x0_normalized, t_horizon, objective_parameters, koop_model)

    # de-normalize control
    control = control * norm_vals[-1]

    full_control = np.zeros([6, ])
    full_control[indices[0]] = control[0]

    if indices[0] == 5:
        full_control[0] = control[1]
        full_control[4] = control[2]
    elif indices[0] == 0:
        full_control[1] = control[1]
        full_control[5] = control[2]
    else:
        full_control[np.max(indices[1:3])] = control[1]
        full_control[np.min(indices[1:3])] = control[2]

    # function returns mpc_x_output and traj_normalized after de-normalizing
    return full_control, mpc_x_output*norm_vals, traj_normalized.T*norm_vals, obj, wse


def control_law_LL_MPC(z, p, t_horizon, objective_parameters):

    theta = z[2]
    theta_dot = z[11]

    x0 = np.zeros([12, ]) #[x y, theta, psiA, psiB, psiF, xdot, ydot, thetadot, psiAdot, psiBdot, psiFdot]
    x0[1] = z[1]  # copy over y-coord
    x0[2] = np.mod(theta, np.pi / 3)  # cast theta into 0<= theta <= pi/3
    x0[6:9] = z[9:12]  # copy over xdot, ydot, thetadot

    z_tie_breaker = z.copy()
    z_tie_breaker[0] = 0

    keypoints = keypoints_spokes_rimless_wheel(z_tie_breaker, p)
    indices = np.argsort(keypoints[1, :])[:3]

    x0[3] = z[3 + indices[0]]  # pseudo psi_A
    x0[3+6] = z[12 + indices[0]]  # pseudo psi_A_dot

    if indices[0] == 5:
        x0[4] = z[3 + 0]  # pseudo psi_B
        x0[5] = z[3 + 4]  # pseudo psi_F
        x0[4 + 6] = z[12 + 0]  # pseudo psi_B_dot
        x0[5 + 6] = z[12 + 4]  # pseudo psi_F_dot
        index_B = 0
        index_F = 4
    elif indices[0] == 0:
        x0[4] = z[3 + 1]  # pseudo psi_B
        x0[5] = z[3 + 5]  # pseudo psi_F
        x0[4 + 6] = z[12 + 1]  # pseudo psi_B_dot
        x0[5 + 6] = z[12 + 5]  # pseudo psi_F_dot
        index_B = 1
        index_F = 5
    else:
        x0[4] = z[3 + np.max(indices[1:3])]  # pseudo psi_B
        x0[5] = z[3 + np.min(indices[1:3])]  # pseudo psi_F
        x0[4 + 6] = z[12 + np.max(indices[1:3])]  # pseudo psi_B_dot
        x0[5 + 6] = z[12 + np.min(indices[1:3])]  # pseudo psi_F_dot
        index_B = np.max(indices[1:3])
        index_F = np.min(indices[1:3])

    if keypoints[0, indices[0]] < 0:
        x0[2] = x0[2] - np.pi/3

    # Determine what contact dynamics to use for local linearization
    index_A = indices[0]

    # Check contact status
    in_contact_A = keypoints[1, index_A] < 0
    in_contact_B = keypoints[1, index_B] < 0
    in_contact_F = keypoints[1, index_F] < 0

    # Dispatch based on contact configuration
    u = np.zeros((3,))
    if in_contact_A and in_contact_B and not in_contact_F:
        C = ddq_AB(x0, p, u)
        A1 = ddq_AB_q(x0, p)
        A2 = ddq_AB_dq(x0, p)
        B = ddq_AB_u(x0, p)
    elif in_contact_A and in_contact_F and not in_contact_B:
        C = ddq_AF(x0, p, u)
        A1 = ddq_AF_q(x0, p)
        A2 = ddq_AF_dq(x0, p)
        B = ddq_AF_u(x0, p)
    elif in_contact_A and not in_contact_B and not in_contact_F:
        C = ddq_A(x0, p, u)
        A1 = ddq_A_q(x0, p)
        A2 = ddq_A_dq(x0, p)
        B = ddq_A_u(x0, p)
    elif not in_contact_A and not in_contact_B and not in_contact_F:
        C = ddq_0(x0, p, u)
        A1 = ddq_0_q(x0, p)
        A2 = ddq_0_dq(x0, p)
        B = ddq_0_u(x0, p)
    else:
        raise ValueError("Unhandled contact case")

    control, mpc_x_output, traj, obj, wse = mpc_LL(x0, t_horizon, objective_parameters, C, A1, A2, B)

    full_control = np.zeros([6, ])
    full_control[indices[0]] = control[0]

    if indices[0] == 5:
        full_control[0] = control[1]
        full_control[4] = control[2]
    elif indices[0] == 0:
        full_control[1] = control[1]
        full_control[5] = control[2]
    else:
        full_control[np.max(indices[1:3])] = control[1]
        full_control[np.min(indices[1:3])] = control[2]

    return full_control, mpc_x_output, traj, obj, wse


def control_law_passive(z, b, Q, u_MPC, u_params):

    # keeps passive spokes at zero velocity
    A_qdd_passive = b + Q

    phis = z[3:9]
    phi_dots = z[-6:]

    max_disp = u_params[0]
    max_speed = u_params[1]

    phi_bools = np.heaviside(max_disp - np.abs(phis), 0)  # 0 when actuator limit is hit
    phi_dot_bools = np.heaviside(max_speed - np.abs(phi_dots), 0)  # 0 when actuator limit is hit

    k_v = 0.2
    u = np.copy(u_MPC)

    for j in range(6):
        if u[j] == 0: # only implement this when u_MPC is NOT controlling a spoke
            if phi_bools[j] == 1 and phi_dot_bools[j] == 1:
                u[j] = -A_qdd_passive[3 + j] - k_v * phi_dots[j]
            else:
                dir1 = np.sign(phis[j]) * (1 - phi_bools[j])  # if 0, no active constraint
                dir2 = np.sign(phi_dots[j]) * (1 - phi_dot_bools[j])  # if 0, no active constraint

                u[j] = -A_qdd_passive[3 + j]
                feedback = -k_v * phi_dots[j]

                if np.sign(feedback) != dir1 and np.sign(feedback) != dir2:
                    u[j] += feedback

    return u

def enforce_bounds(z_out, max_disp, max_speed):

    max_disp = max_disp*0.97
    max_speed = max_speed*0.97

    z_out_new = np.copy(z_out)

    # enforce actuator bounds
    for idx in range(3, 9):
        if np.abs(z_out[idx]) > max_disp:
            z_out_new[idx] = np.sign(z_out[idx]) * max_disp

    for idx in range(12, 18):
        if np.abs(z_out[idx]) > max_speed:
            z_out_new[idx] = np.sign(z_out[idx]) * max_speed

    return z_out_new


def rand_num(a, b):
    return np.random.uniform(a, b)

def initialize_state(p):

    # only used with random initializations
    x = 0
    y = rand_num(0.375, 0.5)
    th = rand_num(-np.pi / 3, np.pi / 3)
    dx = rand_num(-0.5, 0.5)
    dy = rand_num(-0.6, 0.6)
    dth = rand_num(-4, 4)

    phiA = rand_num(-0.075, 0.075)
    phiB = rand_num(-0.075, 0.075)
    phiC = rand_num(-0.075, 0.075)
    phiD = rand_num(-0.075, 0.075)
    phiE = rand_num(-0.075, 0.075)
    phiF = rand_num(-0.075, 0.075)

    dphiA = rand_num(-0.160, 0.160)
    dphiB = rand_num(-0.160, 0.160)
    dphiC = rand_num(-0.160, 0.160)
    dphiD = rand_num(-0.160, 0.160)
    dphiE = rand_num(-0.160, 0.160)
    dphiF = rand_num(-0.160, 0.160)

    r = p[8]
    R = p[9]

    psiA = phiA / (R * r)
    psiB = phiB / (R * r)
    psiC = phiC / (R * r)
    psiD = phiD / (R * r)
    psiE = phiE / (R * r)
    psiF = phiF / (R * r)

    dpsiA = dphiA / (R * r)
    dpsiB = dphiB / (R * r)
    dpsiC = dphiC / (R * r)
    dpsiD = dphiD / (R * r)
    dpsiE = dphiE / (R * r)
    dpsiF = dphiF / (R * r)

    z0 = np.array([x, y, th, psiA, psiB, psiC, psiD, psiE, psiF, dx, dy, dth, dpsiA, dpsiB, dpsiC, dpsiD, dpsiE, dpsiF])
    return z0

if __name__ == "__main__":

    # SIMULATION PARAMETERS --------------------------------------------------------------------------------------------
    l = 0.35
    M = 25
    m = 0.25
    g = 9.81
    I = 1.4518
    ramp = 0 * (np.pi / 180)
    k = 0
    l0 = 0.0750
    r = 0.02/(2*np.pi)
    R = 1/7
    Il = 0.0000847
    Im = 0.00115
    p = np.array([M, m, I, l, g, ramp, k, l0, r, R, Il, Im])  # parameters
    yC = 0

    dt = 0.001
    tf = 5  # LENGTH OF SIMULATION
    num_steps = int(tf / dt)
    tspan = np.linspace(0, tf, num_steps)

    # PASSIVE CONTROL PARAMETERS ---------------------------------------------------------------------------------------
    max_speed = 0.160/(R*r)
    max_disp = 0.075/(R*r)
    noise = 0   # not in use
    u_params = np.array([max_disp, max_speed, noise])


    # KOOPMAN MPC PARAMETERS -------------------------------------------------------------------------------------------
    dt_k = 0.01 # Koopman time step, not necessarily the same as the simulation time step
    t_horizon = 20 # MPC TIME HORIZON, NOT DURATION OF SIM

    # keep track of MPC outputs
    MPC_start_time = 0  # no control inputs for the first x seconds
    MPC_freq = 1 # save one in every n MPC calcs

    # NORMALIZATION PARAMETERS -----------------------------------------------------------------------------------------
    scale_thetadot = 1
    scale_xdot = 1
    scale_theta = 1

    char_length = 2 * l0 + 2 * l
    time_constant = 10 * dt_k
    char_vel = char_length / time_constant

    norm_vals = np.array(
        [char_length, char_length, scale_theta*np.pi, 0.075 / (R * r), 0.075 / (R * r), 0.075 / (R * r), scale_xdot*char_vel, char_vel,
         scale_thetadot*np.pi / time_constant, 0.160 / (R * r), 0.160 / (R * r), 0.160 / (R * r)])

    # LOAD MODEL TYPE---------------------------------------------------------------------------------------------------
    model_type = 'CCK' # CCK, DMDc, or LL
    embedding_compensation = 1 # only applicable for CCK, 1 is on, 0 is off
    initialization = 'trajectory' # trajectory, stance, or random

    # load trajectory files
    normalized_trajectory_file = './../../Data_Rimless_Wheel/reference_trajectories/20deg_normalized_traj_LA36_0.01dt.csv' # three actuator model
    trajectory_file = './../../Data_Rimless_Wheel/reference_trajectories/20deg_traj_LA36_0.01dt.csv' #full state model
    init_conds = pd.read_csv(trajectory_file, header=None)
    init_conds = init_conds.values
    init_conds[0, :] = 0*init_conds[0, :]

    if model_type == 'CCK':
        A_matrix = np.load(r"./../../Data_Rimless_Wheel/koopman_models/cck/A_2000_pass_tb_LA36_0.01dt.npy")
        A_matrix = sparse.csc_matrix(A_matrix)
        B_matrix = np.load(r"./../../Data_Rimless_Wheel/koopman_models/cck/B_2000_pass_tb_LA36_0.01dt.npy")
        B_matrix = sparse.csc_matrix(B_matrix)
        centers = np.load(r"./../../Data_Rimless_Wheel/koopman_models/cck/centers_2000_pass_tb_LA36_0.01dt.npy")
        epsilon = 2

        koop_model = [A_matrix, B_matrix, centers, epsilon, embedding_compensation]

    elif model_type == 'DMDc':
        A_matrix = np.load(r"./../../Data_Rimless_Wheel/koopman_models/dmdc/A_2000_pass_act_tb_LA36_0.01dt_CCK.npy")
        A_matrix = sparse.csc_matrix(A_matrix)
        B_matrix = np.load(r"./../../Data_Rimless_Wheel/koopman_models/dmdc/B_2000_pass_act_tb_LA36_0.01dt_CCK.npy")
        B_matrix = sparse.csc_matrix(B_matrix)
        centers = np.load(r"./../../Data_Rimless_Wheel/koopman_models/dmdc/centers_2000_pass_act_tb_LA36_0.01dt_CCK.npy")
        epsilon = 1.5
        embedding_compensation = 0 # cannot use embedding compensation in DMDc model

        koop_model = [A_matrix, B_matrix, centers, epsilon, embedding_compensation]


    elif model_type == "LL":

        # uses non-normalized data, so different Q, R, and reference trajectory
        Q_mat = 0.0 * np.eye(12)
        Q_mat[0][0] = 2 / pow(norm_vals[0], 2)    # x
        Q_mat[6][6] = 10 / pow(norm_vals[6], 2)   # xdot
        R_mat = np.eye(3) * (1/(351.85)) / pow(norm_vals[-1], 2)

        objective_parameters = ["forwards", trajectory_file, Q_mat, R_mat]

    if model_type == "CCK" or model_type == "DMDc":

        Q_mat = 0.01 * np.eye(B_matrix.shape[0]+1)
        Q_mat[0][0] = 2        #x
        Q_mat[1][1] = 0        #y
        Q_mat[2][2] = 0
        Q_mat[3][3] = 0
        Q_mat[4][4] = 0
        Q_mat[5][5] = 0
        Q_mat[6][6] = 10        #xdot
        Q_mat[7][7] = 0         #ydot
        Q_mat[8][8] = 0         #thetadot
        Q_mat[9][9] = 0
        Q_mat[10][10] = 0
        Q_mat[11][11] = 0

        R_mat = np.eye(3) * 1/(351.85)
        objective_parameters = ["forwards", normalized_trajectory_file, Q_mat, R_mat]


    #  START SIM -------------------------------------------------------------------------------------------------------
    for_loop_start = time.time()
    start_val = (1+1*5)
    total_sims = 1
    end_val = start_val + total_sims - 1


    for sim_iter in range(start_val, start_val + total_sims, 1): # start_val to total_sims - 1
        print(f"----------------------------------------------------------------------------------------")
        print(f"----------------------------------------------------------------------------------------")
        print(f"-------------------Starting simulation {sim_iter-start_val} out of {total_sims}-------------------")

        if start_val != sim_iter:
            for_loop_current = time.time()
            time_per_sim = (for_loop_current-for_loop_start)/(60*(sim_iter-start_val))
            print(f"-----Estimated remaining time for all sims is T-{time_per_sim*(total_sims-(sim_iter-start_val))} min------")


        # INITIAL CONDITION...

        if initialization == 'trajectory':
            z0 = init_conds[:, sim_iter]

        elif initialization == 'stance':
            z0 = np.array(
                [0.1, 0.5, np.pi/6, 0.00, 0.00, 0.0, 0.0, 0.0, 0.00, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            MPC_start_time = 0.3 # let it settle

        elif initialization == 'random':
            np.random.seed(sim_iter)
            z0 = initialize_state(p)

        t_mpc = np.arange(MPC_start_time, tf, MPC_freq * dt_k, dtype=float)
        z_mpc = np.zeros((len(t_mpc) * (t_horizon + 1), 12))
        traj_mpc = np.zeros((len(t_mpc) * (t_horizon + 1), 12))
        mpc_counter = 0

        z_out = np.zeros((18, num_steps))
        z_out[:, 0] = z0
        u_out = np.zeros((6, num_steps))
        u_MPC_out = np.zeros((6, num_steps))

        start = time.time()
        u_MPC = np.zeros((6, 1))


        # Perform dynamic simulation
        for i in range(num_steps - 1):

            if np.mod(i, 10) == 0 and i >= round(MPC_start_time /0.001):
                # MPC is only called every 0.01 seconds

                if model_type == 'LL':
                    u_MPC, MPC_output, traj, obj, wse = control_law_LL_MPC(z_out[:, i], p, t_horizon, objective_parameters)
                else:
                    u_MPC, MPC_output, traj, obj, wse = control_law_MPC(z_out[:, i], p, t_horizon, koop_model,
                                                                        norm_vals,
                                                                        objective_parameters)

                if np.mod(i, 100) == 0 and i != 0:
                    print(f"-------Current simulation is {100 * i / num_steps:.1f}% done-------")
                    predict_end = time.time()

                    remaining_run_time= (num_steps-i) *(predict_end - start)/(60*i)
                    print(f"Estimated remaining run time for this sim is T-{remaining_run_time:.2f} min")

                if i in np.round(t_mpc/dt):
                    z_mpc[mpc_counter * (t_horizon + 1):(mpc_counter + 1) * (t_horizon + 1), :] = MPC_output
                    traj_mpc[mpc_counter * (t_horizon + 1):(mpc_counter + 1) * (t_horizon + 1), :] = traj
                    mpc_counter += 1

            dz, forces, u = dynamics(z_out[:, i], p, yC, u_MPC)

            z_out[:, i + 1] = z_out[:, i] + dz * dt
            z_out[:9, i + 1] = z_out[:9, i] + z_out[9:, i + 1] * dt

            z_out[:, i + 1] = enforce_bounds(z_out[:, i + 1], max_disp, max_speed)
            u_out[:, i + 1] = u.reshape(6,)
            u_MPC_out[:, i + 1] = u_MPC.reshape(6, )


        end = time.time()
        print("this simulation's total elapsed time was:", (end - start) / 60)


        # SAVE STATES AND CONTROL COMMANDS FROM SIM --------------------------------------------------------------------
        np.savetxt(r"./../../Data_Rimless_Wheel/sim_results/z_out/z_out.csv", z_out, delimiter=",")
        np.savetxt(r"./../../Data_Rimless_Wheel/sim_results/z_out/u_out.csv", u_out, delimiter=",")
        np.savetxt(r"./../../Data_Rimless_Wheel/sim_results/z_out/u_MPC_out.csv", u_MPC_out, delimiter=",")

        # SAVE MPC OUTPUTS  --------------------------------------------------------------------------------------------
        np.savetxt(r"./../../Data_Rimless_Wheel/sim_results/MPC_out/z_out_MPC.csv", z_mpc, delimiter=",")
        np.savetxt(r"./../../Data_Rimless_Wheel/sim_results/MPC_out/t_out_MPC.csv", t_mpc, delimiter=",")    # vector of times when MPC is called
        np.savetxt(r"./../../Data_Rimless_Wheel/sim_results/MPC_out/traj_MPC.csv", traj_mpc, delimiter=",")  # reference trajectory of given MPC call


    #   PLOT STATES  ---------------------------------------------------------------------------------------------------
    tspan = np.linspace(0, tf, num_steps)
    fig, axs = plt.subplots(4, 1, figsize=(15, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    axs[0].set_title(f"Actuated Rimless Wheel States vs. Time")
    axs[0].plot(tspan, z_out[0, :], label='x-pos')
    axs[0].plot(tspan, z_out[1, :], label='y-pos')
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel("Position [m]")

    axs[1].plot(tspan, z_out[9, :], '--', label=r'x-vel')
    axs[1].plot(tspan, z_out[10, :], '--', label=r'y-vel')
    axs[1].legend(loc='upper right')
    axs[1].set_ylabel("Velocity [m/s]")

    axs[2].plot(tspan, z_out[3, :], label='psi_A', color="#0072BD")
    axs[2].plot(tspan, z_out[4, :], label='psi_B', color="#D95319")
    axs[2].plot(tspan, z_out[5, :], label='psi_C', color="#EDB120")
    axs[2].plot(tspan, z_out[6, :], label='psi_D', color="#7E2F8E")
    axs[2].plot(tspan, z_out[7, :], label='psi_E', color="#77AC30")
    axs[2].plot(tspan, z_out[8, :], label='psi_F', color="#4DBEEE")
    axs[2].legend(loc='upper right')
    axs[2].set_ylabel("phi [m]")

    axs[3].plot(tspan, u_out[0, :], label='u_psi_A', alpha=0.7, color="#0072BD")
    axs[3].plot(tspan, u_out[1, :], label='u_psi_B', alpha=0.7, color="#D95319")
    axs[3].plot(tspan, u_out[2, :], label='u_psi_C', alpha=0.7, color="#EDB120")
    axs[3].plot(tspan, u_out[3, :], label='u_psi_D', alpha=0.7, color="#7E2F8E")
    axs[3].plot(tspan, u_out[4, :], label='u_psi_E', alpha=0.7, color="#77AC30")
    axs[3].plot(tspan, u_out[5, :], label='u_psi_F', alpha=0.6, color="#4DBEEE")
    axs[3].set_xlabel("Time [s]")
    axs[3].set_ylabel("control effort [Nm]")

    plt.show()
