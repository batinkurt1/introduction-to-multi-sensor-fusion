function [estimated_states,estimated_covariances] = batch_KF(measurements,A,B,C,H,Q,R,t,x0_bar,P0)

    estimated_state = x0_bar;
    estimated_covariance = P0;
    estimated_states = zeros(4,t);
    estimated_covariances = cell(1,t);

    for k = 1:length(measurements)
        [predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(estimated_state,estimated_covariance,A,B,C,Q);

        [estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,measurements(:,k),C,H,R);
        estimated_states(:,k) = estimated_state;
        estimated_covariances{k} = estimated_covariance;
    end
end