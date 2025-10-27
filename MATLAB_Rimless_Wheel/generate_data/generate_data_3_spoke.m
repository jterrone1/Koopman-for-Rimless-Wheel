addpath("./rimless_scripts_3")

% set up data generation parameters
N = 100; % number of trajectories (around 2000)
E_max = 250; 
dt = 0.01; %IMPORTANT PARAMETER
tf = 51*dt; % [MAX] duration of each trajectory
ground_percent = 5; % usually 3. Put neg number if you don't care
collision_increase = 3; % at minimum, n/10 of sims must start near pi/6, integers only. Put neg number if you don't care

% LINAK  LA36
l = 350*10^-3; % 200+STROKE LENGTH = 200 + 150 = 350 mm closed length
l0 = 75*10^-3; % full stroke length is 150 mm
M = 4*6+1; % actuator mass is 4 Kg x 6 + 1 Kg for the housing
m = 0.25; % mass at spoke tip estimated as 0.25 kg
k = 0;
g = 9.81;
I = 3*(1/12 *((2*4))*(2*(l+l0))^2) + 0.0034*2; % "rod rotating about center", inertia is I = 1/12 m l^2 + housing inertia
r = 0.02/(2*pi);  % worm gear has a 20 mm spindle pitch (20 mm per full rotation) [m/rad]
R = 1/7; % gear ratio is 1:7
Il = 8.47 * 10^-5;
Im = 1.15*10^-3;

ramp = 0 * (pi/180);
p   = [M m I l g ramp k l0 r R Il Im]';

% ground contact parameters
yC = 0;
K_z = 1.0e2;
D_z = 5.0e3;
l_max = 0.1;
fr_coeff = 0.8;

pg = [yC K_z D_z l_max fr_coeff]; 

X_t = [];
X_tp1_act = [];
X_tp1_pass = [];
X_tp1_pass2 = [];
u_t = [];
z0_save = [];

for n = 1:1:N

    flag = 1;

    while flag == 1

        z0 = init_conds(E_max, p, pg, n, ground_percent);
        [z_out, z_out_pass, u_out, flag] = simulate_rimless_fn(p, pg, z0, dt, tf);

    end
    
    z0_save = [z0_save, z0];

    X_t = [X_t; z_out(:, 1:end-1)'];
    u_t = [u_t; u_out'];
    
    
    X_tp1_act = [X_tp1_act; z_out(:, 2:end)'];
    X_tp1_pass = [X_tp1_pass; z_out_pass(:, 2:end)'];

    if mod(n, floor(N/100)) == 0
        fprintf('%d%% completed \n', 100*n/N)
    end

end

writematrix(X_t,"../../Data_Rimless_Wheel/training/X_t_" + N + "_pass_tb_LA36_" + dt+"dt.csv") 
writematrix(X_tp1_pass,"../../Data_Rimless_Wheel/training/X_tp1_" + N + "_pass_tb_LA36_" + dt+"dt.csv") 
%writematrix(X_tp1_act,"../../Data_Rimless_Wheel/training/X_tp1_" + N + "_act_tb_LA36_" + dt+"dt.csv") 
%writematrix(u_t, "../../Data_Rimless_Wheel/training/inputs_" + N + "_act_tb_LA36_" + dt+"dt.csv") 

%% Functions

function z0 = init_conds(E_max, p, pg, n, ground_percent)

    E_curr = 300;
    ground_contact = 1;

    while E_curr > E_max | ground_contact == 0

        if (mod(n, 10) < ground_percent)
            ground_contact = 0;
        end


        % LINAK
        x = 0;
        y = rand_num(0.375, 0.5, 1);      
        th = rand_num(-pi/3, pi/3, 1);      %[-pi/3, pi/3]
        dx = rand_num(-0.5, 0.5, 1);      %[-0.5, 0.5]
        dy = rand_num(-0.6, 0.6, 1);      %[-0.6, 0.6]
        dth = rand_num(-4, 4, 1);       %[-4, 4]

        phiA = rand_num(-0.075, 0.075, 1);
        phiB = rand_num(-0.075, 0.075, 1);
        phiF = rand_num(-0.075, 0.075, 1);

        dphiA = rand_num(-0.160, 0.160, 1); %no load, 160 mm/s or with load 135 mm/s
        dphiB = rand_num(-0.160, 0.160, 1);
        dphiF = rand_num(-0.160, 0.160, 1);

        r = p(9);
        R = p(10);

        psiA = phiA/(R*r);
        psiB = phiB/(R*r);
        psiF = phiF/(R*r);

        dpsiA = dphiA/(R*r);
        dpsiB = dphiB/(R*r);
        dpsiF = dphiF/(R*r);

        z0 = [x; y; th; psiA; psiB; psiF; dx; dy; dth; dpsiA; dpsiB; dpsiF];

        gr_energy = ground_energy(z0, p, pg);
        energy = energy_rimless_wheel_3_spoke(z0, p);

        if gr_energy > 0
            ground_contact = 1;
        end

        E_curr = energy + gr_energy;
    end

end

function [z_out, z_out_pass, u_out, flag] = simulate_rimless_fn(p, pg, z0, dt, tf)


    %% Perform Dynamic simulation    
    num_steps = floor(tf/dt);
    tspan = linspace(0, tf, num_steps); 
    z_out = zeros(12,num_steps);
    z_out(:,1) = z0;
    z_out_pass = zeros(12,num_steps);
    z_out_pass(:,1) = z0;
    u_out = zeros(3,num_steps-1);

    r = p(9);
    R = p(10);

    for i=1:num_steps-1
        [dz, u, flag] = dynamics(z_out(:,i), p, pg, 1); %actuated
        [dz_pass, u_pass, flag_pass] = dynamics(z_out(:,i), p, pg, 0); %passive

        if flag == 1 %check if forces are large
            i;
            break
        end

        if flag == 2 %stop data collection right before spokes C, D, E touch the ground
            if i <= 3 && tf/dt > 3
                flag = 1;
                break
            end
            z_out = z_out(:, 1:i);
            z_out_pass = z_out_pass(:, 1:i);
            u_out = u_out(:, 1:i-1);
            break
        end

        z_out(:,i+1) = z_out(:,i) + dz*dt;
        z_out_pass(:,i+1) = z_out(:,i) + dz_pass*dt;

        u_out(:, i) = u;

        if max(abs(z_out(4:6,i+1))) > 0.075/(R*r)
            z_out = z_out(:, 1:i);
            z_out_pass = z_out_pass(:, 1:i);
            u_out = u_out(:, 1:i-1);
            if i <= 3 && tf/dt > 3
                flag = 1;
            end
            break
        end

        if max(abs(z_out(1:2,i+1))) > 0.6 % stop if flying away
            flag = 1;
            break
        end

        if max(abs(z_out(2,i+1))) < 0.25 % stop if sinking into the floor
            flag = 1;
            break
        end

    end
    
end

function [dz, u, flag] = dynamics(z, p, pg, control_bool)


    % Get mass matrix
    A = A_rimless_wheel_3_spoke(z,p);
    
    % Get forces
    b = b_rimless_wheel_3_spoke(z, p);

    forces = contact_force(z, p, pg);

    flag = 0;

    if  max(max(abs(forces))) > 10e4 %check here
        flag = 1;
    end

    if max(max(abs(forces(:, 3:5)))) > 0
        flag = 2;
    end

    B = [0, 0, 0;
    0, 0, 0;
    0, 0, 0;
    1, 0, 0;
    0, 1, 0;
    0, 0, 1;];

    QF_A = jacobian_A_rimless_wheel_3_spoke(z,p)' * forces(:,1);
    QF_B = jacobian_B_rimless_wheel_3_spoke(z,p)' * forces(:,2);
    QF_F = jacobian_F_rimless_wheel_3_spoke(z,p)' * forces(:,6);

    Q = QF_A + QF_B + QF_F;


    % max/min noise = 4 Nm
    if control_bool
        r = p(9);
        R = p(10);
        u = control_law(z, b, Q, 0.075/(R*r), 0.160/(R*r), 4);
    else
        u = zeros(3,1);
    end

    % Solve for qdd
    qdd = pinv(A)*(b + Q + B*u);
    dz = 0*z;
    
    % Form dz
    dz(1:6) = z(7:end);
    dz(7:end) = qdd;

end

function forces = contact_force(z, p, pg)
        keypoints = keypoints_rimless_wheel_3_spoke(z,p);
        keypoints_vel = keypoints_vel_rimless_wheel_3_spoke(z,p);

        forces = zeros(2,6);

        yC =        pg(1);
        K_z =       pg(2);
        D_z  =      pg(3);
        l_max =     pg(4);
        fr_coeff =  pg(5);

        for i = 1:1:length(keypoints)

            rN = keypoints(:,i);
            C_l = rN(2) - yC;

            if C_l < 0
                
                dC_l = keypoints_vel(:,i);

                F_N = -K_z*tan(pi*C_l/(2*l_max)) - D_z*abs(C_l)*dC_l(2);
                F_f = -(2/pi)*atan(dC_l(1)/0.001)*fr_coeff*F_N;

                forces(:,i) = [F_f; F_N];

            end
        end
end

function energy = ground_energy(z, p, pg)
        keypoints = keypoints_rimless_wheel_3_spoke(z,p);
        keypoints_vel = keypoints_vel_rimless_wheel_3_spoke(z,p);

        yC =        pg(1);
        K_z =       pg(2);
        D_z  =      pg(3);
        l_max =     pg(4);
        fr_coeff =  pg(5);

        energy = 0;
        
        for i = 1:1:length(keypoints)

            rN = keypoints(:,i);
            C_l = rN(2) - yC;

            if C_l < 0
                
                energy = energy + 0.5*K_z*(abs(C_l)^2);

            end
        end
end

function r = rand_num(a, b, n)
    r = a + (b-a)*rand(n,1);
end

function u = control_law(z, b, Q, max_disp, max_speed, noise)
    
    A_qdd_passive = b + Q;

    phis = z(4:6);
    phi_dots = z(end-2:end);
    phi_bools = floor(heaviside(max_disp - abs(phis))); % 0 when actuator limit is hit
    phi_dot_bools = floor(heaviside(max_speed - abs(phi_dots))); % 0 when actuator limit is hit

    u = zeros(3,1);
    k_v = 0.1;

    for j=1:1:3
        if phi_bools(j) == 1 && phi_dot_bools(j) == 1
            u(j) = -A_qdd_passive(3 + j) - k_v*phi_dots(j) + rand_num(-noise, noise, 1);
        else
            dir = sign(phis(j))*(1-phi_bools(j)); % if 0, no active constraint
            dir2 = sign(phi_dots(j))*(1-phi_dot_bools(j)); % if 0, no active constraint

            u(j) = -A_qdd_passive(3 + j);
            
            feedback = - k_v*phi_dots(j);
            noise_val = rand_num(-noise, noise, 1);

            if (sign(feedback + noise_val) ~= dir && sign(feedback + noise_val) ~= dir2)
                u(j) = u(j) + feedback + noise_val;
            elseif (sign(feedback) ~= dir && sign(feedback) ~= dir2)
                u(j) = u(j) + feedback;
            end
        end
    end 

end

