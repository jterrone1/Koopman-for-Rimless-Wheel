% plots animation for three spoke model (such as MPC rollout data)

addpath("\..\generate_data\rimless_scripts_3\")
filename = ".\..\..\Data_Rimless_Wheel\sim_results\MPC_out\z_out_MPC_31_traj_CCK_B_p20.csv";

t_horizon = 20;

z_out = csvread(filename)';

%% Define fixed paramters

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
p   = [M m I l g ramp k l0 r R Il Im]';       % parameters

range_x = [min(z_out(1,:))-3*l,max(z_out(1,:))+3*l];

range_x = [-0.5, 0.6];
x_diff = diff(range_x);

yC = 0;


%% Simple Animation for Three Spokes

tspan = 0:0.001:(0.001*length(z_out));

[~, name, ~] = fileparts(filename);  
myVideo = VideoWriter('test_MPC', 'MPEG-4'); 
myVideo.FrameRate = 10;
open(myVideo)

R = [cos(-ramp), -sin(-ramp); sin(-ramp), cos(-ramp)];

limit1 = R*[range_x(1); yC];
limit2 = R*[range_x(2); yC];

figure(2); clf;


hold on
spoke_AD = plot([0],[0],'LineWidth',2, 'Color',"k");
spoke_BE = plot([0],[0],'LineWidth',2, 'Color',"k");
spoke_CF = plot([0],[0],'LineWidth',2,'Color',"k");

spring_A = plot([0],[0],'LineWidth',2, 'Color', "#0072BD"); % dark blue [0]
spring_B = plot([0],[0],'LineWidth',2, 'Color', "#D95319"); % orange [1]
spring_C = plot([0],[0],'LineWidth',2, 'Color', "#EDB120"); % yellow [2]
spring_D = plot([0],[0],'LineWidth',2, 'Color', "#7E2F8E"); % purple [3]
spring_E = plot([0],[0],'LineWidth',2, 'Color', "#77AC30"); % green [4]
spring_F = plot([0],[0],'LineWidth',2, 'Color', "#4DBEEE"); % light blue [5]

spring_tip_A = scatter([-10],[0], 'Color', "k");
spring_tip_B = scatter([-10],[0], 'Color', "k");
spring_tip_F = scatter([-10],[0], 'Color', "k");

xlabel('x [m]')
ylabel('y [m]');
h_title = title('t = 0.0');

axis equal
axis_x1 = min(limit1(1), limit2(1));
axis_x2 = max(limit1(1), limit2(1));
axis_y1 = min(limit1(2), limit2(2))-2*l;
axis_y2 = max(limit1(2), limit2(2))+5*l;

axis_y1 = min(limit1(2), limit2(2))-1*l;
axis_y2 = max(limit1(2), limit2(2))+3*l;

% modify here to change axis
axis([axis_x1 axis_x2 axis_y1 axis_y2]);

line([limit1(1),limit2(1)],[limit1(2),limit2(2)])

actuated = find(z_out(13:end, 1));

%%
%Step through and update animation
for i=1:length(z_out)

    
    if mod(i, t_horizon+1) == 1 && i > 1
       pause(1)  % put a break point here to pause between MPC rollouts
    end
    
    % interpolate to get state at current time.
    t = tspan(i);
    z = z_out(:,i);
    keypoints = keypoints_rimless_wheel_mass(z,p);
    spokes = keypoints_spokes_rimless_wheel_mass(z,p);

    rA = R*keypoints(:,1);
    rB = R*keypoints(:,2);
    rC = R*keypoints(:,3); 
    rD = R*keypoints(:,4);
    rE = R*keypoints(:,5);
    rF = R*keypoints(:,6); 

    rA_sp = R*spokes(:,1);
    rB_sp = R*spokes(:,2);
    rC_sp = R*spokes(:,3); 
    rD_sp = R*spokes(:,4);
    rE_sp = R*spokes(:,5);
    rF_sp = R*spokes(:,6); 


    set(h_title,'String',  sprintf('MPC Rollout') ); % update title
    

    % Plot Pole
    set(spoke_AD,'XData' , [rA_sp(1) rD_sp(1)] );
    set(spoke_AD,'YData' , [rA_sp(2) rD_sp(2)] );

    set(spoke_BE,'XData' , [rB_sp(1) rE_sp(1)] );
    set(spoke_BE,'YData' , [rB_sp(2) rE_sp(2)] );

    set(spoke_CF, 'XData' , [rC_sp(1) rF_sp(1)] );
    set(spoke_CF, 'YData' , [rC_sp(2) rF_sp(2)] );

    set(spring_A, 'XData' , [rA_sp(1) rA(1)] );
    set(spring_A, 'YData' , [rA_sp(2) rA(2)] );

    set(spring_B, 'XData' , [rB_sp(1) rB(1)] );
    set(spring_B, 'YData' , [rB_sp(2) rB(2)] );

    set(spring_C, 'XData' , [rC_sp(1) rC(1)] );
    set(spring_C, 'YData' , [rC_sp(2) rC(2)] );

    set(spring_D, 'XData' , [rD_sp(1) rD(1)] );
    set(spring_D, 'YData' , [rD_sp(2) rD(2)] );

    set(spring_E, 'XData' , [rE_sp(1) rE(1)] );
    set(spring_E, 'YData' , [rE_sp(2) rE(2)] );

    set(spring_F, 'XData' , [rF_sp(1) rF(1)] );
    set(spring_F, 'YData' , [rF_sp(2) rF(2)] );

    frame = getframe(gcf); %get frame
    writeVideo(myVideo, frame);

    pause(0.001)
end



close(myVideo)
