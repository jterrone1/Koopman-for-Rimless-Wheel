function lambda = contact_force(z, p)
        keypoints = keypoints_cartpole(z,p);
        rP = keypoints(:,2);

        k = p(5);

        if rP(1) < -0.1
            lambda2 = max(0, k*(-0.1-rP(1)));
            lambda1 = 0;
        elseif rP(1) > 0.1
            lambda1 = max(0, -k*(-rP(1) + 0.1));
            lambda2 = 0;
        else
            lambda2 = 0;
            lambda1 = 0;
        end

        lambda = [lambda1, lambda2]';

end