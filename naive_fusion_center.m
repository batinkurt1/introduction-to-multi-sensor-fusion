function [estimated_states,estimated_covariances] = naive_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0)
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
        fused_covariance = inv(inv(S1_estimated_covariance)+inv(S2_estimated_covariance));
        fused_state = fused_covariance * inv(S1_estimated_covariance) * S1_estimated_state + fused_covariance * inv(S2_estimated_covariance) * S2_estimated_state;
        S2_estimated_state = fused_state;
        S2_estimated_covariance = fused_covariance;
    end
    estimated_states(:,k) = S2_estimated_state;
    estimated_covariances{k} = S2_estimated_covariance;
end

end