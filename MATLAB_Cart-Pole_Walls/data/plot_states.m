addpath('.\plotting');

zout2 = load("klqr_rng5.csv");
zout1 = load("lqr_rng5.csv");

zout1 = zout1(:, 1:end-100);
zout2 = zout2(:, 1:end-100);

uout2 = load("u_klqr_rng5.csv");
uout1 = load("u_lqr_rng5.csv");

uout1 = uout1(:, 1:end-100);
uout2 = uout2(:, 1:end-100);



num_steps = size(zout1, 2);       
dt = 0.01;                    
tspan = (0:num_steps-1) * dt;     
systems = { 'LQR', ...
            'KLQR' };

cmap = [0, 0.4470, 0.7410;    % blue
        0.8500, 0.3250, 0.0980; % orange
        0.4660, 0.6740, 0.1880; % greenish 
        0.4940, 0.1840, 0.5560; % purple
        0.3010, 0.7450, 0.9330]; % cyan



ax(1) = subplot(3,2,1);
hold on;
plot(tspan, zout1(1,:), 'LineWidth',1, 'DisplayName','$x$ [m]', 'Color', cmap(1,:));
plot(tspan, zout1(3,:), 'LineWidth',1, 'DisplayName','$\dot{x}$ [m/s]', 'Color', cmap(2,:));
hold off;
ylabel('$\mathbf{x}$ states', 'Interpreter', 'latex');
title(systems{1});

ax(4) = subplot(3,2,3);
hold on;
plot(tspan, zout1(2,:), 'LineWidth',1, 'DisplayName','$\theta$ [rad]', 'Color', cmap(3,:));
plot(tspan, zout1(4,:), 'LineWidth',1, 'DisplayName','$\dot{\theta}$ [rad/s]', 'Color', cmap(4,:));
hold off;
ylabel('$\theta$ states', 'Interpreter', 'latex');

ax(7) = subplot(3,2,5);
plot(tspan, uout1, 'LineWidth',1, 'DisplayName','$u$ [N]', 'Color', cmap(5,:));
ylabel('Control [N]', 'Interpreter', 'latex');
xlabel('Time [s]');


ax(2) = subplot(3,2,2);
hold on;
plot(tspan, zout2(1,:), 'LineWidth',1, 'DisplayName','$x$ [m]', 'Color', cmap(1,:));
plot(tspan, zout2(3,:), 'LineWidth',1, 'DisplayName','$\dot{x}$ [m/s]', 'Color', cmap(2,:));
hold off;
title(systems{2});
legend('show','Interpreter','latex','Location','best','Box','off'); 

ax(5) = subplot(3,2,4);
hold on;
plot(tspan, zout2(2,:), 'LineWidth',1, 'DisplayName','$\theta$ [rad]', 'Color', cmap(3,:));
plot(tspan, zout2(4,:), 'LineWidth',1, 'DisplayName','$\dot{\theta}$ [rad/s]', 'Color', cmap(4,:));
hold off;
legend('show','Interpreter','latex','Location','best','Box','off');

ax(8) = subplot(3,2,6);
plot(tspan, uout2, 'LineWidth',1, 'DisplayName','$u$ [N]', 'Color', cmap(5,:));
xlabel('Time [s]');
legend('show','Interpreter','latex','Location','best','Box','off');

linkaxes([ax(1) ax(2)], 'y'); 
linkaxes([ax(4) ax(5)], 'y'); 
linkaxes([ax(7) ax(8)], 'y'); 
