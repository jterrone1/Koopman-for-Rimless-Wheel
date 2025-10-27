function [koop_model, flag] = generate_lqr(koop_model, Q, R)

A = koop_model.A;
B = koop_model.B;

try
    [K, ~, ~] = dlqr(A, B, Q, R);
    flag = 1; % dlqr succeeded
catch
    K = zeros(size(A));
    flag = 0; % dlqr failed
end

koop_model.K = K;
koop_model.Q = Q;
koop_model.R = R;


end

