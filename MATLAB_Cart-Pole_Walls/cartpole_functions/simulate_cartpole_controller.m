function [z_out, u_out] = simulate_cartpole_controller(koop_model, rng_val)

    %% Define fixed paramters
    
    mP = 0.1;
    mC = 1;
    l = 0.5;
    g = 9.81;
    k = 10;
    p   = [mP mC l g k]';        % parameters
    %rng(0)


    %% Perform Dynamic simulation    
    dt = 0.01;
    tf = 6;
    num_steps = floor(tf/dt);
    tspan = linspace(0, tf, num_steps); 
    rng(rng_val)
    z0 = [0.2*(rand(1)-0.5); 0*(rand(1)-0.5); 3*(rand(1)-0.5); 2*(rand(1)-0.5)];

    z_out = zeros(4,num_steps);
    z_out(:,1) = z0;

    u_out = zeros(1,num_steps);
    for i=1:num_steps-1
        
        [dz, u, flag] = dynamics(z_out(:,i), p, 2, koop_model, 0);

        z_out(:,i+1) = z_out(:,i) + dz*dt;
        u_out(:, i) = u;
       

    end
    

end