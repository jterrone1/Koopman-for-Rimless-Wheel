clear
clc

addpath('.\cartpole_functions');
addpath('.\koopman_models');

mP = 0.1;
mC = 1;
l = 0.5;
g = 9.81;
k = 10;
p   = [mP mC l g k]'; 



%%  TEST CONTROLLERS

controller_values = [1, 2];         % two controllers
num_tests = 10;                     % number of tests total

% load in koopman model/klqr controller 
load('obs_25_eps_18.1.mat', 'best_model');
koop_model = best_model;

counter1_1 = 0; % # koopman trials with avg MSE < 0.01 in last second of sim
counter1_2 = 0; % # koopman trials with avg MSE < 0.001 in last second of sim

       
for j = 1:1:num_tests
    [z_out, u_out] = simulate_cartpole_controller(koop_model, j);

    if sum(mean(z_out(1:4, end-100:end).^2, 2))/4 < 0.01
        counter1_1 = counter1_1 + 1;
    end
    if sum(mean(z_out(1:4, end-100:end).^2, 2))/4 < 0.001
        counter1_2 = counter1_2 + 1;
    end

    if mod(j,100) == 0
        j/num_tests
    end
end

counter2_1 = 0; % # lqr trials with avg MSE < 0.01 in last second of sim
counter2_2 = 0; % # lqr trials with avg MSE < 0.001 in last second of sim

for j = 1:1:num_tests
    [z_out, u_out] = simulate_cartpole_controller_LQR(j);

    if sum(mean(z_out(1:4, end-100:end).^2, 2))/4 < 0.01
        counter2_1 = counter2_1 + 1;
    end
    if sum(mean(z_out(1:4, end-100:end).^2, 2))/4 < 0.001
        counter2_2 = counter2_2 + 1;
    end

    if mod(j,100) == 0
        j/num_tests
    end
end
