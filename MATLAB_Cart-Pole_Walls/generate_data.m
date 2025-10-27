clear
clc

addpath('.\cartpole_functions');

% set up data generation parameters
N = 10; % number of trajectories (around 2000)
dt = 0.01; %IMPORTANT PARAMETER
tf = 60*dt; % [MAX] duration of each trajectory

mP = 0.1;
mC = 1;
l = 0.5;
g = 9.81;
k = 10;
p   = [mP mC l g k]';        % parameters

X_t = [];
X_tp1_act = [];
X_tp1_pass = [];
u_t = [];
z0_save = [];


for n = 1:1:N

    z0 = [0.2*(rand(1)-0.5); 0; 2*(rand(1)-0.5); 2*(rand(1)-0.5)];
    [z_out, z_out_pass, u_out, flag] = simulate_cartpole(p, z0, dt, tf);


    X_t = [X_t; z_out(:, 1:end-1)'];
    u_t = [u_t; u_out'];
    
    X_tp1_act = [X_tp1_act; z_out(:, 2:end)'];
    X_tp1_pass = [X_tp1_pass; z_out_pass(:, 2:end)'];

    if mod(n, floor(N/100)) == 0
        fprintf('%d%% completed \n', 100*n/N)
    end

end

writematrix(X_t,"./data/training/X_t_" + N + "_pass_cartpole0_" + dt+"dt.csv") 
writematrix(X_tp1_pass,"./data/training/X_tp1_" + N + "_pass_cartpole0_" + dt+"dt.csv") 

%writematrix(X_t,"./data/training/X_t_" + N + "_act_cartpole1_" + dt+"dt.csv") 
%writematrix(X_tp1_act,"./data/training/X_tp1_" + N + "_act_cartpole1_" + dt+"dt.csv") 
%writematrix(u_t, "./data/training/inputs_" + N + "_act_cartpole1_" + dt+"dt.csv") 


%% Functions

function [z_out, z_out_pass, u_out, flag] = simulate_cartpole(p, z0, dt, tf)


    num_steps = floor(tf/dt);
    tspan = linspace(0, tf, num_steps); 
    z_out = zeros(4,num_steps);
    z_out(:,1) = z0;
    z_out_pass = zeros(4,num_steps);
    z_out_pass(:,1) = z0;
    u_out = zeros(1,num_steps-1);


    for i=1:num_steps-1
        [dz, u, flag] = dynamics(z_out(:,i), p, 1); %actuated
        [dz_pass, u_pass, flag_pass] = dynamics(z_out(:,i), p, 0); %passive

        if flag == 1 %check if forces are high
            i;
            break
        end

        z_out(:,i+1) = z_out(:,i) + dz*dt;
        z_out_pass(:,i+1) = z_out(:,i) + dz_pass*dt;

        u_out(:, i) = u;

        if max(abs(z_out(1,i+1))) > 1 % stop if flying away
            flag = 1;
            break
        end

    end
    
end


function r = rand_num(a, b, n)
    r = a + (b-a)*rand(n,1);
end

