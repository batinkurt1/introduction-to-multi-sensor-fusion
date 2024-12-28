clc;
clearvars;
close all;

T = 1;
A = [eye(2),T*eye(2);0*eye(2),eye(2)];
B = [T^2/2*eye(2);T*eye(2)];
C = [eye(2),0*eye(2)];
H = eye(2);
t = 100;

x0_bar = [5000,5000,25,25]';
P0 = diag((x0_bar/10).*(x0_bar/10));

process_noise_std = 2;
Q = diag([process_noise_std^2,process_noise_std^2]);
process_noise_mean = [0;0];


measurement_noise_std = 20;
R = diag([measurement_noise_std^2,measurement_noise_std^2]);
measurement_noise_mean = [0;0];


trueTarget = zeros(4,t);

trueTarget(:,1) = mvnrnd(x0_bar,P0)';

for k = 2:t
    process_noise = mvnrnd(process_noise_mean,Q)';
    trueTarget(:,k) = A * trueTarget(:,k-1) + B * process_noise;
end



S1_measurements = zeros(2,t);
S2_measurements = zeros(2,t);

for k = 1:t
    S1_measurement_noise = mvnrnd(measurement_noise_mean,R)';
    S2_measurement_noise = mvnrnd(measurement_noise_mean,R)';
    S1_measurements(:,k) = C * trueTarget(:,k) + H * S1_measurement_noise;
    S2_measurements(:,k) = C * trueTarget(:,k) + H * S2_measurement_noise;
end


% Centralized Solution

C_augmented = [C;C];
R_augmented = [R,zeros(2,2);zeros(2,2),R];
H_augmented = [H,zeros(2,2);zeros(2,2),H];

augmented_measurements = zeros(4,t);

for k = 1:t
    augmented_measurements(:,k) = [S1_measurements(:,k);S2_measurements(:,k)];
end

[centralized_estimates,centralized_covariances] = batch_KF(augmented_measurements,A,B,C_augmented,H_augmented,Q,R_augmented,t,x0_bar,P0);

% what happens as number of fusion centers goes to infinity?

[naive_estimated_states,naive_estimated_covariances] = naive_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);

[channel_filter_estimated_states,channel_filter_estimated_covariances] = channel_filter_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);

[LEA_estimated_states,LEA_estimated_covariances] = LEA_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);

[CI_estimated_states,CI_estimated_covariances] = covariance_intersection_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);

figure;
plot(trueTarget(1,:),trueTarget(2,:),LineWidth=1.5,Color="#77AC30");
hold on;
plot(centralized_estimates(1,:),centralized_estimates(2,:),LineWidth=1.5,Color="#D95319");
title("True Target Trajectory vs. Fusion Algorithms");
ylabel("y position");
xlabel("x position");
plot(naive_estimated_states(1,:),naive_estimated_states(2,:),LineWidth=1.5);
plot(channel_filter_estimated_states(1,:),channel_filter_estimated_states(2,:),LineWidth=1.5);
plot(LEA_estimated_states(1,:),LEA_estimated_states(2,:),LineWidth=1.5);
plot(CI_estimated_states(1,:),CI_estimated_states(2,:),LineWidth=1.5);

legend("True Target Trajectory","Centralized Fusion","Naive Fusion","Channel Filter","LEA Fusion","CI Fusion");
grid on;

Nmc = 100;
NEESmc_centralized = zeros(Nmc,t);
NEESmc_naive = zeros(Nmc,t);
NEESmc_channel_filter = zeros(Nmc,t);
NEESmc_LEA = zeros(Nmc,t);
NEESmc_CI = zeros(Nmc,t);

for i = 1:Nmc
    trueTarget = zeros(4,t);
    trueTarget(:,1) = mvnrnd(x0_bar,P0)';

    for k = 2:t
        process_noise = mvnrnd(process_noise_mean,Q)';
        trueTarget(:,k) = A * trueTarget(:,k-1) + B * process_noise;
    end


    S1_measurements = zeros(2,t);
    S2_measurements = zeros(2,t);
    
    for k = 1:t
        S1_measurement_noise = mvnrnd(measurement_noise_mean,R)';
        S2_measurement_noise = mvnrnd(measurement_noise_mean,R)';
        S1_measurements(:,k) = C * trueTarget(:,k) + H * S1_measurement_noise;
        S2_measurements(:,k) = C * trueTarget(:,k) + H * S2_measurement_noise;
    end
    

    augmented_measurements = zeros(4,t);
    
    for k = 1:t
        augmented_measurements(:,k) = [S1_measurements(:,k);S2_measurements(:,k)];
    end
    
    [centralized_estimates,centralized_covariances] = batch_KF(augmented_measurements,A,B,C_augmented,H_augmented,Q,R_augmented,t,x0_bar,P0);
    
    [naive_estimated_states,naive_estimated_covariances] = naive_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);
    
    [channel_filter_estimated_states,channel_filter_estimated_covariances] = channel_filter_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);
    
    [LEA_estimated_states,LEA_estimated_covariances] = LEA_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);
    
    [CI_estimated_states,CI_estimated_covariances] = covariance_intersection_fusion_center(S1_measurements,S2_measurements,A,B,C,H,Q,R,t,x0_bar,P0);

    for k = 1:t
        NEESmc_centralized(i,k) = (trueTarget(:,k) - centralized_estimates(:,k))' * inv(centralized_covariances{k}) * (trueTarget(:,k) - centralized_estimates(:,k));
        NEESmc_naive(i,k) = (trueTarget(:,k) - naive_estimated_states(:,k))' * inv(naive_estimated_covariances{k}) * (trueTarget(:,k) - naive_estimated_states(:,k));
        NEESmc_channel_filter(i,k) = (trueTarget(:,k) - channel_filter_estimated_states(:,k))' * inv(channel_filter_estimated_covariances{k}) * (trueTarget(:,k) - channel_filter_estimated_states(:,k));
        NEESmc_LEA(i,k) = (trueTarget(:,k) - LEA_estimated_states(:,k))' * inv(LEA_estimated_covariances{k}) * (trueTarget(:,k) - LEA_estimated_states(:,k));
        NEESmc_CI(i,k) = (trueTarget(:,k) - CI_estimated_states(:,k))' * inv(CI_estimated_covariances{k}) * (trueTarget(:,k) - CI_estimated_states(:,k));
    end

end

ANEES_centralized = 1/Nmc*sum(NEESmc_centralized);

ANEES_naive = 1/Nmc*sum(NEESmc_naive);

ANEES_channel_filter = 1/Nmc*sum(NEESmc_channel_filter);

ANEES_LEA = 1/Nmc*sum(NEESmc_LEA);

ANEES_CI = 1/Nmc*sum(NEESmc_CI);

gamma_min = chi2inv(0.005, Nmc*4);
gamma_max = chi2inv(1 - 0.005, Nmc*4);
lower_threshold = gamma_min/Nmc;
upper_threshold = gamma_max/Nmc;

figure;
plot(1:t,ANEES_centralized);
grid on;
hold on;
plot(1:t,ANEES_naive);
plot(1:t,ANEES_channel_filter);
plot(1:t,ANEES_LEA);
plot(1:t,ANEES_CI);
yline(lower_threshold);
yline(upper_threshold);
title("ANEES");

legend("Centralized Fusion","Naive Fusion","Channel Filter","LEA Fusion","CI Fusion");
grid on;

centralized_position_estimation_error = sqrt((trueTarget(1,:)-centralized_estimates(1,:)).^2+(trueTarget(2,:)-centralized_estimates(2,:)).^2);
centralized_rms_position_error = sqrt(1/t*(sum(centralized_position_estimation_error.^2)));
centralized_velocity_estimation_error = sqrt((trueTarget(3,:)-centralized_estimates(3,:)).^2+(trueTarget(4,:)-centralized_estimates(4,:)).^2);
centralized_rms_velocity_error = sqrt(1/t*(sum(centralized_velocity_estimation_error.^2)));

naive_position_estimation_error = sqrt((trueTarget(1,:)-naive_estimated_states(1,:)).^2+(trueTarget(2,:)-naive_estimated_states(2,:)).^2);
naive_rms_position_error = sqrt(1/t*(sum(naive_position_estimation_error.^2)));
naive_velocity_estimation_error = sqrt((trueTarget(3,:)-naive_estimated_states(3,:)).^2+(trueTarget(4,:)-naive_estimated_states(4,:)).^2);
naive_rms_velocity_error = sqrt(1/t*(sum(naive_velocity_estimation_error.^2)));

channel_filter_position_estimation_error = sqrt((trueTarget(1,:)-channel_filter_estimated_states(1,:)).^2+(trueTarget(2,:)-channel_filter_estimated_states(2,:)).^2);
channel_filter_rms_position_error = sqrt(1/t*(sum(channel_filter_position_estimation_error.^2)));
channel_filter_velocity_estimation_error = sqrt((trueTarget(3,:)-channel_filter_estimated_states(3,:)).^2+(trueTarget(4,:)-channel_filter_estimated_states(4,:)).^2);
channel_filter_rms_velocity_error = sqrt(1/t*(sum(channel_filter_velocity_estimation_error.^2)));

LEA_position_estimation_error = sqrt((trueTarget(1,:)-LEA_estimated_states(1,:)).^2+(trueTarget(2,:)-LEA_estimated_states(2,:)).^2);
LEA_rms_position_error = sqrt(1/t*(sum(LEA_position_estimation_error.^2)));
LEA_velocity_estimation_error = sqrt((trueTarget(3,:)-LEA_estimated_states(3,:)).^2+(trueTarget(4,:)-LEA_estimated_states(4,:)).^2);
LEA_rms_velocity_error = sqrt(1/t*(sum(LEA_velocity_estimation_error.^2)));

CI_position_estimation_error = sqrt((trueTarget(1,:)-CI_estimated_states(1,:)).^2+(trueTarget(2,:)-CI_estimated_states(2,:)).^2);
CI_rms_position_error = sqrt(1/t*(sum(CI_position_estimation_error.^2)));
CI_velocity_estimation_error = sqrt((trueTarget(3,:)-CI_estimated_states(3,:)).^2+(trueTarget(4,:)-CI_estimated_states(4,:)).^2);
CI_rms_velocity_error = sqrt(1/t*(sum(CI_velocity_estimation_error.^2)));

fprintf("Position RMS of Centralized Fusion: %0.5g \n",centralized_rms_position_error);
fprintf("Velocity RMS of Centralized Fusion: %0.5g \n",centralized_rms_velocity_error);
fprintf("Position RMS of Naive Fusion: %0.5g \n",naive_rms_position_error);
fprintf("Velocity RMS of Naive Fusion: %0.5g \n",naive_rms_velocity_error);
fprintf("Position RMS of Channel Filter: %0.5g \n",channel_filter_rms_position_error);
fprintf("Velocity RMS of Channel Filter: %0.5g \n",channel_filter_rms_velocity_error);
fprintf("Position RMS of LEA Fusion: %0.5g \n",LEA_rms_position_error);
fprintf("Velocity RMS of LEA Fusion: %0.5g \n",LEA_rms_velocity_error);
fprintf("Position RMS of CI Fusion: %0.5g \n",CI_rms_position_error);
fprintf("Velocity RMS of CI Fusion: %0.5g \n",CI_rms_velocity_error);

