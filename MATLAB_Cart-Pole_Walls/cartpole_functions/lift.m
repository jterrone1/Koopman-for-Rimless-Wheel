function y = lift(centers, epsilon, X)

    observable_vals = RBF(centers, epsilon, X);
    y = [X, observable_vals]; % lifted state 

    % y = [w(1), q(1), g1(1), g2(1)...gn(1);
    %      w(2), q(2), g1(2), g2(2)...gn(2);
    %       :     :      :       :      :
    %      w(N), q(N), g1(N), g2(N)...gn(N)] 

end