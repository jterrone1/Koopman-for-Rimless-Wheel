function koop_model = generate_koopman(X_t, X_tp1, n, epsilon)

[~, centers] = kmeans(X_tp1, n);

yt = lift(centers, epsilon, X_t)'; % lift data sets
ytp1 = lift(centers, epsilon, X_tp1)';
    
    % yt = [w(1),  w(2),  w(3) ... w(N);
    %       q(1),  q(2),  q(3) ... q(N);
    %       g1(1), g1(2), g1(3)...g1(N);
    %         :     :      :        :
    %       gn(1), gn(2), gn(3)...gn(N)] 
    
A = ytp1 * pinv(yt); % least squares solution, (n + 4) x (n + 4)


% currently, these parameters are hardcoded in!!!
dt = 0.01;
mC = 1;
l = 0.5;

B = zeros(length(A), 1);
B(3) = dt/mC;
B(4) = dt/(l*mC);

koop_model.A = A;
koop_model.B = B;
koop_model.centers = centers;
koop_model.epsilon = epsilon;

end