function [estimated_states,estimated_covariances] = LEA_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0)
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
        [U1,Lambda1,V1] = svd(S1_estimated_covariance);
        Tau_1 = sqrtm(Lambda1)\U1';
        P_C = Tau_1 * S2_estimated_covariance * Tau_1';
        [U2,Lambda2,V2] = svd(P_C);
        Tau_2 = U2' * Tau_1;
        z_1 = Tau_2 * S1_estimated_state;
        z_2 = Tau_2 * S2_estimated_state;

        Z_1 = Tau_2 * S1_estimated_covariance * Tau_2';
        Z_2 = Tau_2 * S2_estimated_covariance * Tau_2';

        indices = [];
        for i = 1:size(Z_2,1)
            if Z_2(i,i) < 1
                indices = [indices, i];
            end
        end

        z_fused_state_estimate = zeros(size(z_1));
        z_fused_covariance_estimate = zeros(size(Z_2)); 

        for i = 1:size(z_1)
            if ismember(i,indices)
                z_fused_state_estimate(i) = z_2(i);
                z_fused_covariance_estimate(i,i) = Z_2(i,i);
            else
                z_fused_state_estimate(i) = z_1(i);
                z_fused_covariance_estimate(i,i) = Z_1(i,i);
            end
        end
        fused_state = Tau_2 \ z_fused_state_estimate;
        fused_covariance = Tau_2 \ z_fused_covariance_estimate * inv(Tau_2');
        S2_estimated_state = fused_state;
    S2_estimated_covariance = fused_covariance;
    end

    estimated_states(:,k) = S2_estimated_state;
    estimated_covariances{k} = S2_estimated_covariance;    
end
end
