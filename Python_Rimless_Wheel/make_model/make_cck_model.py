# %% import modules 
import numpy as np
from numpy.linalg import norm, inv
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from numpy import linalg as LA
import seaborn as sns

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

# k-means++
def kmeans_calc(X, obs_dim, vis=False):
    km = KMeans(n_clusters=obs_dim, init='k-means++',
                n_init='auto', max_iter=300,
                tol=1e-04, random_state=0, copy_x=False)
    clusters_sklearn = km.fit_predict(X[1:12, :].T)
    centers = km.cluster_centers_
    #np.save("./hardware_based/centers_a15_p8_100mm_100obs_eps40.npy", centers)
    if vis == True:
        plt.plot(X[0, :], X[1, :], 'ro', ms=0.4, alpha=1, label='Ground Truth')
        plt.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                'kx', label='RBF Centers')
        plt.xlabel("X-Pos [m]")
        plt.ylabel("Y-Pos [m]")
        plt.title("Trajectory Data for Actuated Rimless Wheel COM")
        plt.legend(fontsize=14)
        plt.show()
    else:
        pass
    return centers

# %% EDMD calculation
def edmd(X, Y, centers, b_val, dt, lam, eps):
    X_lift = X[1:12, :].T # remove x data
    Y_lift = Y[1:12, :].T
    for c in centers:
        X_lift = np.concatenate([X_lift, RBF(c, X[1:12, :].T, eps).reshape(-1, 1)], axis=1)
        Y_lift = np.concatenate([Y_lift, RBF(c, Y[1:12, :].T, eps).reshape(-1, 1)], axis=1)
    X_lift = X_lift.T
    Y_lift = Y_lift.T

    Regularized_Y_lift = np.hstack((Y_lift, np.zeros((np.shape(X_lift)[0], np.shape(X_lift)[0]))))
    Regularized_X_lift = np.hstack((X_lift, np.shape(X_lift)[1]*lam*np.identity(np.shape(X_lift)[0])))

    A_lift = Regularized_Y_lift@LA.pinv(Regularized_X_lift)

    B_lift = np.zeros((A_lift.shape[0], 3)) # when x is removed, first line is Y
    B_lift[8][0] = b_val              # corresponds to psidotA
    B_lift[9][1] = b_val              # corresponds to psidotB
    B_lift[10][2] = b_val             # corresponds to psidotF



    np.save(r"./../../Data_Rimless_Wheel/koopman_models/cck/A_test.npy", A_lift)
    np.save(r"./../../Data_Rimless_Wheel/koopman_models/cck/B_test.npy", B_lift)
    np.save(r"./../../Data_Rimless_Wheel/koopman_models/cck/centers_test.npy", centers)

    return A_lift, B_lift



def error_1dt(csv_xt, csv_xtp1, u_bool, u, A, B, centers, dt, eps, norm_vals = None):

    xt = pd.read_csv(csv_xt, header=None)
    data_t = xt.values

    xtp1 = pd.read_csv(csv_xtp1, header=None)
    data_tp1 = xtp1.values

    if norm_vals is not None:
        [data_t, data_tp1] = normalize_data(data_t.T, data_tp1.T, norm_vals)
        data_t = data_t.T
        data_tp1 = data_tp1.T

    error = 0
    error_not_normalized = 0

    MSE_per_state = np.zeros(np.shape(xt)[1]-1)

    if u_bool:
        u_inputs = pd.read_csv(u, header=None)
        u_vals = u_inputs.values

    for i in range(len(data_t)):
        x_current = data_t[i, 1:12] #excluding x-data
        Z = x_current

        for c in centers:
            rbf = RBF(c, x_current.reshape(1, -1), eps)
            Z = np.concatenate([Z, rbf], axis=0)

        Z_out = A@Z
        if u_bool:
            Z_out += B@u_vals[i, :].T

        error += (LA.norm(Z_out[5:11] - data_tp1[i, 6:12])) ** 2 # VELOCITY MSE
        MSE_per_state += np.square(Z_out[0:11] - data_tp1[i, 1:12])

        z_pred, z_actual = undo_normalize_data(Z_out[0:11], data_tp1[i, 1:12], norm_vals[1:])
        error_not_normalized += (LA.norm(z_pred[5:11] - z_actual[5:11])) ** 2 # NON-NORMALIZED VELOCITY MSE

    error_not_normalized = error_not_normalized/len(data_t)
    return error/len(data_t), MSE_per_state/len(data_t)

def auto_epsilon(X, Y, centers, dt, eps_linspace, comparisonX, comparisonY, comparisonU=None, norm_vals=None):

    best_MSE = 10000
    best_eps = 10000
    A_lift_store = []
    B_lift_store = []

    eps_MSE = []

    if comparisonU is None:
        u_bool = False
    else:
        u_bool = True

    for eps in eps_linspace:
        print(eps)
        A_lift, B_lift = edmd(X, Y, centers, (Il * pow(R, 2) + Im), dt, 0, eps)  # alpha used for ridge regularization
        MSE1, MSE_vec1 = error_1dt(comparisonX, comparisonY, u_bool, comparisonU, A_lift, B_lift, centers, dt, eps, norm_vals)
        eps_MSE.append(MSE1)

        if MSE1 < best_MSE:
            best_MSE = MSE1
            best_eps = eps
            A_lift_store = A_lift
            B_lift_store = B_lift

    plt.plot(eps_linspace, np.log10(eps_MSE))
    plt.title(f"Epsilon vs MSE for 100 RBF")
    plt.xlabel("Epsilon")
    plt.ylabel("log of MSE")
    plt.show()

    return best_eps, A_lift_store, B_lift_store

def normalize_data(X, Y, norm_vals):
    # normalize dataset for LS, returns new data
    norm_vals = norm_vals.reshape(len(norm_vals), 1)
    X_normalized = X/norm_vals
    Y_normalized = Y/norm_vals
    return X_normalized, Y_normalized

def undo_normalize_data(X, Y, norm_vals):
    # normalize dataset for LS, returns new data
    norm_vals = norm_vals.reshape(len(norm_vals), 1)

    X_not_normalized = np.multiply(X.reshape(len(norm_vals), 1), norm_vals)
    Y_not_normalized = np.multiply(Y.reshape(len(norm_vals), 1), norm_vals)
    return X_not_normalized, Y_not_normalized

# %%
# if __name__ == "__main__":

# trajectory based with stabilizing controller, N = 168,529
xt = pd.read_csv('./../../Data_Rimless_Wheel/training/X_t_2000_pass_tb_LA36_0.01dt.csv', header=None)
xt1 = pd.read_csv('./../../Data_Rimless_Wheel/training/X_tp1_2000_pass_tb_LA36_0.01dt.csv', header=None)
X_true = xt.values.T
Y_true = xt1.values.T


print(f"The current dataset has {np.shape(X_true)[1]} samples")

state_dim = 12
dt = 0.01

# LINAK  LA36
r = 0.02/(2*np.pi)  # lead screw has diameter 46 mm --> radius = 0.023 m
R = 1/7 # gear ratio is 1:7
Il = 0.0000847
Im = 0.00115
l0 = 0.075
l = 0.35

char_length = 2*l0 + 2*l
time_constant = 10*dt
char_vel = char_length/time_constant

scale_theta = 1
scale_thetadot = 1
scale_xdot = 1

norm_vals = np.array([char_length, char_length, scale_theta*np.pi, 0.075/(R*r), 0.075/(R*r), 0.075/(R*r), scale_xdot*char_vel, char_vel,
                      scale_thetadot*np.pi/time_constant, 0.160/(R*r), 0.160/(R*r), 0.160/(R*r)])

[X, Y] = normalize_data(X_true, Y_true, norm_vals)

obs_dim = 100
lift_dim = state_dim + obs_dim
centers = kmeans_calc(X, obs_dim)

csv_xt = './../../Data_Rimless_Wheel/training/comparison/X_t_50_pass_tb_LA36_0.01dt_comparison.csv'
csv_xtp1 = './../../Data_Rimless_Wheel/training/comparison/X_tp1_50_pass_tb_LA36_0.01dt_comparison.csv'
csv_u = ''

#eps_linspace = np.linspace(0.01,12, 24)
#[best_eps, A_lift, B_lift] = auto_epsilon(X, Y, centers, dt, eps_linspace, csv_xt, csv_xtp1, norm_vals = norm_vals)

best_eps = 2
A_lift, B_lift = edmd(X, Y, centers, dt/(Il*pow(R,2) + Im), dt, 0, best_eps) # lambda used for ridge regularization
# model is saved to Data_Rimless_Wheel/koopman_models/cck/