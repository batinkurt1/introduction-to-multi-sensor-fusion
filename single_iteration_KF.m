function [estimated_state,estimated_covariance] = single_iteration_KF(measurement,A,B,C,H,Q,R,estimated_state,estimated_covariance)

[predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(estimated_state,estimated_covariance,A,B,C,Q);

[estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,measurement,C,H,R);
end