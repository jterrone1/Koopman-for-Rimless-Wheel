function [dz, u, flag] = dynamics(z, p, controller_type, varargin, varargin2)

    if nargin == 4
        koop_model = varargin;
        fp_input = 0;
    elseif nargin > 4
        koop_model = varargin;
        fp_input = varargin2;
    else
        koop_model = [];
        fp_input = 0;
    end

    % Get mass matrix
    A = A_cartpole(z,p);
    
    b = b_cartpole(z, p);

    lambda = contact_force(z, p);

    forces = [-lambda(1); 0] + [lambda(2); 0];
    QF_P = jacobian_P(z,p)' * forces;

    flag = 0;

    if  max(max(abs(forces))) > 10e4 %check here
        flag = 1;
    end

    if controller_type == 1
        % from Stabilization of Complementarity Systems via Contact-Aware Controllers

        K = [3.6931  -46.7030    3.3885   -5.7147];
        L = [-13.9797   13.9767];
    
        u = K*z + L*lambda;

    
    elseif controller_type == 2
        % KOOPMAN LQR CONTROLLER
        
        if isempty(koop_model)
            fprintf('ERROR: Missing koopman model \n')
        end

        K = koop_model.K;
        centers = koop_model.centers;
        epsilon = koop_model.epsilon;

        z_lifted = lift(centers, epsilon, z')';
    
        u = -K*z_lifted;

    elseif controller_type == 3
        % Regular LQR Controller
        
        K = [10.0000000000000	-91.7719537328621	16.2818544511592	-22.6933137560722];
        u = K*z;

    elseif controller_type == 0
        % NO CONTROL INPUT
        u = zeros(1,1);
    end

    % Solve for qdd
    l = p(3);
    mC = p(2);

    %real control input
    qdd = A\(b + QF_P + [u; 0]);

    % linearized control
    %qdd = A\(b + QF_P);
    %qdd = qdd + [1/mC; 1/(l*mC)]*u;
    dz = 0*z;
    
    % Form dz
    dz(1:2) = z(3:4);
    dz(3:4) = qdd;


end