function simulate_cartpole()

    clf
    clear
    clc

    addpath('.\cartpole_functions');
    addpath('.\koopman_models');

    %% Define fixed paramters
    
    mP = 0.1;
    mC = 1;
    l = 0.5;
    g = 9.81;
    k = 10;
    p   = [mP mC l g k]';        % parameters

    fp_input = 0; %ignore
    controller_type = 2; % 0 = passive, 2 = koopman, 3 = lqr

    % load in koopman model/klqr controller 
    load('obs_25_eps_18.1.mat', 'best_model');
    koop_model = best_model;

    %% Perform Dynamic simulation    
    dt = 0.01;
    tf = 6;
    num_steps = floor(tf/dt);
    tspan = linspace(0, tf, num_steps); 
    rng_val = 5;
    rng(rng_val)

    z0 = [0.2*(rand(1)-0.5); pi/3*(rand(1)-0.5); 2*(rand(1)-0.5); 2*(rand(1)-0.5)];
    z0 = [0.2*(rand(1)-0.5); 0*(rand(1)-0.5); (3)*(rand(1)-0.5); 2*(rand(1)-0.5)];
    
    z_out = zeros(4,num_steps);
    u_out = zeros(1,num_steps);
    z_out(:,1) = z0;
    for i=1:num_steps-1
        
        if fp_input
            [dz, u, flag] = dynamics(z_out(:,i), p, controller_type, koop_model, fp_input);
        else
            [dz, u, flag] = dynamics(z_out(:,i), p, controller_type, koop_model);
        end

        z_out(:,i+1) = z_out(:,i) + dz*dt;
        u_out(:, i) = u;
       
    end
    final_state = z_out(:,end);


    %% Simple Animation

    myVideo = VideoWriter('test'); %open video file
    myVideo.FrameRate = 10; 
    open(myVideo)

    limit1 = [min(z_out(1,:))-3*l; 0];
    limit2 = [max(z_out(1,:))+3*l; 0];

    limit1 = [-3*l; 0];
    limit2 = [3*l; 0];

    figure(2); clf;
    
    hold on
    cart = plot([0],[0],'LineWidth',5);
    pole = plot([0],[0],'LineWidth',2);
    xlabel('x')
    ylabel('y');
    h_title = title('t=0.0s');
    
    axis equal
    skip_frame = 10;


    line([limit1(1),limit2(1)],[limit1(2),limit2(2)])


    line([-0.1 -0.1], [0, 0.7], 'Color','k', 'LineWidth',0.5);
    line([0.1 0.1], [0, 0.7], 'Color','k', 'LineWidth',0.5);

    %Step through and update animation
    for i=1:num_steps
        
        if mod(i, skip_frame)
           continue
        end

        % interpolate to get state at current time.
        t = tspan(i);

        z = z_out(:,i);
        keypoints = keypoints_cartpole(z,p);

        rC = keypoints(:,1);
        rP = keypoints(:,2);

        set(h_title,'String',  sprintf('t=%.2f',t) ); % update title
        

        % Plot Pole
        set(cart,'XData' , [rC(1)-0.05 rC(1)+0.05] );
        set(cart,'YData' , [rC(2) rC(2)] );

        set(pole,'XData' , [rC(1) rP(1)] );
        set(pole,'YData' , [rC(2) rP(2)] );


        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);

        pause(.01)
    end

    close(myVideo)

    clf
    
    
end