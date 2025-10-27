function observable_vals = RBF(centers, epsilon, X)

    observable_vals = [];
    [n, dim] = size(centers);

    for i = 1:1:n
        r = vecnorm(X-centers(i,:), 2, 2); % L2 norm
        RBF_val = exp(-(r/(epsilon)^2));
        % np.exp(-(r/(epsilon))**2) and r = norm(x - c) from python code
        observable_vals = [observable_vals, RBF_val]; 
    end

end