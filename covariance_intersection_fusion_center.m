function [estimated_states,estimated_covariances] = covariance_intersection_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0)
S1_estimated_state = x0_bar;
S1_estimated_covariance = P0;
S2_estimated_state = x0_bar;
S2_estimated_covariance = P0;

estimated_states = zeros(4,t);
estimated_covariances = cell(1,t);
for k = 1:t
    [S1_estimated_state,S1_estimated_covariance] = single_iteration_KF(S1_measurements(:,k),A,B,C,H,Q,R,S1_estimated_state,S1_estimated_covariance);
    [S2_estimated_state,S2_estimated_covariance] = single_iteration_KF(S2_measurements(:,k),A,B,C,H,Q,R,S2_estimated_state,S2_estimated_covariance);
    if mod(k,2) == 1
        ws = linspace(0,1,20);
        w_star = 0;
        smallest_covariance_norm = inf;
        % optimization
        for i = 1:length(ws)
            fused_covariance = inv(ws(i)*inv(S1_estimated_covariance)+(1-ws(i))*inv(S2_estimated_covariance));
            if trace(fused_covariance) < smallest_covariance_norm
                smallest_covariance_norm = trace(fused_covariance);
                w_star = ws(i);
            end
        end
        fused_covariance = inv(w_star*inv(S1_estimated_covariance)+(1-w_star)*inv(S2_estimated_covariance));
        fused_state = fused_covariance * (w_star *inv(S1_estimated_covariance)* S1_estimated_state + (1-w_star)*inv(S2_estimated_covariance) * S2_estimated_state);
        S2_estimated_state = fused_state;
        S2_estimated_covariance = fused_covariance;
    end

    estimated_states(:,k) = S2_estimated_state;
    estimated_covariances{k} = S2_estimated_covariance;    
end
end
