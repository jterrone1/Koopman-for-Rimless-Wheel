import pandas as pd
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    print("Importing gurobipy: Success")
except ModuleNotFoundError:
    print("Importing gurobipy: Fail")


def RBF(center, x, eps, kinds="gauss"):
    """
    This function returns RBF functions based on kmeans++ center positions and the states.
    """
    c = center
    epsilon = eps
    r = norm(x - c, axis=1, ord=2)
    if kinds == "gauss":
        return np.exp(-(r/(epsilon))**2)
    elif kinds == "quad":
        return 1/(1+(r/epsilon)**2)
    elif kinds == "inv_quadric":
        return 1/np.sqrt(1+(epsilon*r)**2)
    elif kinds == "thin":
        return r**2 * np.log(r)
    elif kinds == "quintic":
        return r**5
    elif kinds == "bump":
        func = np.exp(-1/(1-(epsilon*r)**2))
        # If >1/epliron, the values should be zero.
        return func
    else:
        pass

def get_normalized_trajectory(x0_normalized, T, trajectory_file, action):

    th_curr = x0_normalized[2]
    th_dot_curr = x0_normalized[8]

    xt = pd.read_csv(trajectory_file, header=None)

    if action == "forwards":
        X = xt.values
    else:
        print("unknown input")
        X = xt.values

    #X[8, :] = X[8, :] # IF THERE IS THETADOT SCALING, MAKE SURE THIS IS UPDATED ACCORDINGLY
    #X[6,:] = X[6,:] # IF THERE IS XDOT SCALING, MAKE SURE THIS IS UPDATED ACCORDINGLY

    x_traj_thetas = np.copy(X)
    x_traj_thetas[2] = np.mod(x_traj_thetas[2], th_curr)

    errors = [x_traj_thetas[2], x_traj_thetas[8] - th_dot_curr]
    errors = np.linalg.norm(errors, axis=0)

    index_start = np.argmin(errors[:-T])

    traj_all = np.copy(X[:, index_start:(index_start+T+1)])
    traj_all[0, :] = traj_all[0, :] - X[0, index_start]

    theta_offset = np.mod(X[2, index_start], th_curr) + th_curr - X[2, index_start]
    traj_all[2, :] = traj_all[2, :] + theta_offset

    return traj_all


def mpc_cck(x0_normalized, T, objective_parameters, koop_model):

    if not getattr(mpc_cck, "_has_run", False):
        print("First time called, building the model!")
        mpc_cck._has_run = True
        m, u, x, x0_constr, z_constr_arr = mpc_create_model(T, koop_model, objective_parameters)

        mpc_cck.model = [m, u, x, x0_constr, z_constr_arr]

    else:
        print("Updating model...")

    action = objective_parameters[0]

    if action == "forwards":
        trajectory_file = objective_parameters[1]
        traj_normalized = get_normalized_trajectory(x0_normalized, T, trajectory_file, action)
        control, x_out, obj, wse = mpc_update_and_solve(mpc_cck.model, x0_normalized, T, traj_normalized, koop_model, objective_parameters)
    else:
        print("not a valid objective...")
        control = None
        x_out = None
        traj_normalized = None
        obj = None
        wse = None

    return control, x_out, traj_normalized, obj, wse

def mpc_create_model(T, koop_model, objective_parameters):

    m = gp.Model("MPC")
    m.Params.LogToConsole = 0  # make verbose
    m.Params.DualReductions = 0
    m.Params.NumericFocus = 2 # usually 2

    A_temp = koop_model[0]
    B_temp = koop_model[1]
    embedding_compensation_bool = koop_model[4]

    Q = objective_parameters[2]
    R = objective_parameters[3]

    # Incorporate x-pos back into A matrix
    A = np.zeros((A_temp.shape[0]+1, A_temp.shape[1]+1))
    A[0, 0] = 1
    A[0, 6] = 0.01/0.1 # should be dt_k/(char_vel/char_length)
    A[1:, 1:] = A_temp.A

    B = np.zeros((B_temp.shape[0]+1, B_temp.shape[1]))
    B[1:, :] = B_temp.A

    [nx, nu] = B.shape

    if embedding_compensation_bool:
        Agp = A[12:, 9:12]
        App = A[9:12, 9:12]

        B[12:] = Agp @ inv(App) @ B[9:12]

    # create dummy lifted trajectories and initial conditions
    traj_lifted = np.zeros([T+1, nx])
    x0 = np.zeros([nx, ])

    umin = np.array([-1.59, -1.59, -1.59])
    umax = np.array([1.59, 1.59, 1.59])

    xmin = -15 * np.ones([nx, ])
    xmax = 15 * np.ones([nx, ])

    xmin[3:6] = np.array([-1.0, -1.0, -1.0]) # maximum and minimum actuator length
    xmin[9:12] = np.array([-1.0, -1.0, -1.0])  # maximum and minimum actuator speed
    xmax[3:6] = np.array([1.0, 1.0, 1.0])
    xmax[9:12] = np.array([1.0, 1.0, 1.0])

    umin = np.tile(umin, (T, 1))
    umax = np.tile(umax, (T, 1))
    x = m.addMVar(shape=(T + 1, nx), lb=xmin, ub=xmax, name='x')
    z = m.addMVar(shape=(T + 1, nx), lb=-GRB.INFINITY, name='z') # error
    u = m.addMVar(shape=(T, nu), lb=umin, ub=umax, name='u')

    x0_constr = m.addConstr(x[0, :] == x0.flatten())
    z_constr_arr = []

    for k in range(T):
        m.addConstr(x[k + 1, :] == A @ x[k, :] + B @ u[k, :])
        z_constr = m.addConstr(z[k, :] == x[k, :] - traj_lifted[k, :])
        z_constr_arr.append(z_constr)

    z_constr = m.addConstr(z[T, :] == x[T, :] - traj_lifted[T, :])
    z_constr_arr.append(z_constr)
    #m.addConstr(z[T, :] == x[T, :] - traj_lifted[T, :])

    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(T + 1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(T))

    m.setObjective(obj1 + obj2, GRB.MINIMIZE)

    return m, u, x, x0_constr, z_constr_arr

def mpc_update_and_solve(model, x0_normalized, T, traj_normalized, koop_model, objective_parameters):
    m = model[0]
    u = model[1]
    x = model[2]
    x0_constr = model[3]
    z_constr_arr = model[4]

    B_temp = koop_model[1]
    centers = koop_model[2]
    epsilon = koop_model[3]

    # to obtain nx, nu
    B = np.zeros((B_temp.shape[0]+1, B_temp.shape[1]))
    B[1:, :] = B_temp.A

    [nx, nu] = B.shape

    traj_lifted = np.zeros((T+1, nx))
    x0 = np.copy(x0_normalized)

    for t in range(T+1):

        q_ref = traj_normalized[:, t]
        x_ref = np.copy(q_ref)

        for c in centers:
            rbf = RBF(c, q_ref[1:].reshape(1, -1), epsilon)
            x_ref = np.concatenate([x_ref, rbf], axis=0)

        traj_lifted[t, :] = x_ref

    for c in centers:
        rbf = RBF(c, x0_normalized[1:].reshape(1, -1), epsilon)
        x0 = np.concatenate([x0, rbf], axis=0)

    m.update()
    x0_constr.rhs = x0.T
    for ind in range(len(z_constr_arr)):
        z_constr = z_constr_arr[ind]
        z_constr.rhs = -traj_lifted[ind]

    start_time = time.time()
    m.optimize()
    end_time = time.time()
    print(end_time - start_time)
    # pred = x.X[1:2,:]

    opt_status = m.status
    if opt_status == 3:
        print("Infeasible")
        m.computeIIS()
        #m.write('iismodel.ilp')

        for c in m.getConstrs():
            if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

        print('Infeasible, relaxing...')
        m.feasRelaxS(1, False, False, True)
        m.optimize()
        opt_status = m.status
        print('Optimization status:', opt_status)
        if opt_status == 3:
            print('Still Infeasible!')
            1+1

    elif opt_status == 12:
        print("Numerical error!")
        1+1

    Q = objective_parameters[2]
    R = objective_parameters[3]

    if opt_status == 12 or opt_status == 3:
        control = np.array([0, 0, 0])
        opt_states = np.zeros((T+1, 12))
        weighted_state_error = 0.001 * opt_status
        obj_val = 0
    else:
        control = u.X[:1, :][0]
        opt_states = x.X[:, :12]
        state_error = x.X[:, :12] - traj_lifted[:, :12]
        weighted_state_error = sum(state_error[k, :] @ Q[0:12, 0:12] @ state_error[k, :] for k in range(T + 1))
        obj_val = m.objVal

    print(control)

    return control, opt_states, obj_val, weighted_state_error


def mpc_LL(x0, T, objective_parameters, C, A1, A2, B):

    m = gp.Model("MPC")
    m.Params.LogToConsole = 0  # make verbose
    m.Params.DualReductions = 0
    #m.Params.NumericFocus = 2
    #m.Params.Aggregate = 0

    nx = 12
    nu = 3

    trajectory_file = objective_parameters[1]
    action = objective_parameters[0]
    traj= get_normalized_trajectory(x0, T, trajectory_file, action)
    traj = np.vstack((traj[0:6,:], traj[9:15,:])).T

    Q = objective_parameters[2]
    R = objective_parameters[3]

    umin = np.array([-1.59, -1.59, -1.59])*351.858377
    umax = np.array([1.59, 1.59, 1.59])*351.858377

    xmin = -100 * np.ones([nx, ])
    xmin[3:6] = np.array([-1.0, -1.0, -1.0])*164.933614 # maximum and minimum actuator length
    xmin[9:12] = np.array([-1.0, -1.0, -1.0])*351.858377 # maximum and minimum actuator speed
    xmax = 100 * np.ones([nx, ])
    xmax[3:6] = np.array([1.0, 1.0, 1.0])*164.933614
    xmax[9:12] = np.array([1.0, 1.0, 1.0])*351.858377

    umin = np.tile(umin, (T, 1))
    umax = np.tile(umax, (T, 1))
    x = m.addMVar(shape=(T + 1, nx), lb=xmin, ub=xmax, name='x')
    z = m.addMVar(shape=(T + 1, nx), lb=-GRB.INFINITY, name='z') # error
    u = m.addMVar(shape=(T, nu), lb=umin, ub=umax, name='u')

    m.addConstr(x[0, :] == x0.flatten())
    C = C.flatten()

    for k in range(T):
        state_error = x[k, :]-x0
        qddot = C + A1 @ state_error[0:6] + A2 @ state_error[6:12] + B @ u[k, :]

        m.addConstr(x[k + 1, 0:6] == x[k, 0:6] + 0.01 * x[k, 6:12])
        m.addConstr(x[k + 1, 6:12] == x[k, 6:12] + 0.01 * qddot)
        m.addConstr(z[k, :] == x[k, :] - traj[k, :])

    m.addConstr(z[T, :] == x[T, :] - traj[T, :])

    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(T + 1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(T))

    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    m.optimize()


    opt_status = m.status
    if opt_status == 3:
        print("Infeasible")

        m.computeIIS()
        #m.write('iismodel.ilp')

        for c in m.getConstrs():
            if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

        print('Infeasible, relaxing...')
        m.feasRelaxS(1, False, False, True)
        m.optimize()
        opt_status = m.status
        print('Optimization status:', opt_status)
        if opt_status == 3:
            print('Still Infeasible!')
            1+1


    elif opt_status == 12:
        print("Numerical error!")
        1+1

    if opt_status == 12 or opt_status == 3:
        control = np.array([0, 0, 0])
        opt_states = np.zeros((T+1, 12))
        weighted_state_error = 0.001 * opt_status
        obj_val = 0
    else:
        control = u.X[:1, :][0]
        opt_states = x.X[:, :12]
        state_error = x.X[:, :12] - traj[:, :12]
        weighted_state_error = sum(state_error[k, :] @ Q[0:12, 0:12] @ state_error[k, :] for k in range(T + 1))
        obj_val = m.objVal

    return control, opt_states, traj, obj_val, weighted_state_error
