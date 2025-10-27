clear
clc

addpath('.\cartpole_functions');
addpath('.\data');

dt = 0.01;
mP = 0.1;
mC = 1;
l = 0.5;
g = 9.81;
k = 10;
p   = [mP mC l g k]';        % parameters


X_t = load('X_t_2000_pass_cartpole0_0.01dt.csv');
X_tp1 = load('X_tp1_2000_pass_cartpole0_0.01dt.csv'); % assuming passive data

x_weight = 100;
th_weight = 100;
xdot_weight = 100;
thdot_weight = 100;

R = 1; 

%%  TEST A RANGE OF EPSILON VALUES AND NUMBER OF OBSERVABLES FOR CONTROLLER PERFORMANCE

n_values = 25:1:25;        % range of n
eps_values = 0.1:1:20.1;     % range of eps

loss_fn_best = 1000;
loss_fn_results = zeros(length(n_values), length(eps_values)); 


for i = 1:length(n_values)
    n = n_values(i);

    for j = 1:length(eps_values)
        eps = eps_values(j);

        fprintf('Currently on n = %d, epsilon = %.2f\n', n, eps)

        Q_diag =  zeros(n+4, 1);
        Q_diag(1) = x_weight; 
        Q_diag(2) = th_weight; 
        Q_diag(3) = xdot_weight; 
        Q_diag(4) = thdot_weight; 
        Q = diag(Q_diag);
        
        koop_model = generate_koopman(X_t, X_tp1, n, eps);
        [koop_model, flag] = generate_lqr(koop_model, Q, R);

        if flag == 1
            z_out_0 = simulate_cartpole_controller(koop_model, 0);
            z_out_1 = simulate_cartpole_controller(koop_model, 1);
            z_out_2 = simulate_cartpole_controller(koop_model, 2);

            z_total = [z_out_0, z_out_1, z_out_2];
            mean_vals = mean(abs(z_total), 2); % mean absolute value
            mean_vals(isnan(mean_vals)) = 1000; % replace NaNs with 1000

        else
            mean_vals = [1000, 1000, 1000, 1000];
        end
                          
        weighted_mean = [20*mean_vals(1), 10*mean_vals(2), mean_vals(3), mean_vals(4)];
        loss_fn = norm(weighted_mean);
        loss_fn_results(i,j) = loss_fn;

        % check for best
        if (loss_fn < loss_fn_best) 
            koop_model.m_x = mean_vals(1);
            koop_model.m_th = mean_vals(2);
            koop_model.m_xd = mean_vals(3);
            koop_model.m_thd = mean_vals(4);

            best_model = koop_model;
            loss_fn_best = loss_fn;
        end

    end
end

save(fullfile('koopman_models', ...
    ['obs_' num2str(length(best_model.centers)) '_eps_' num2str(best_model.epsilon) '.mat']), 'best_model');
