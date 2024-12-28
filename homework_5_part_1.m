clearvars;
clc;
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


measurements = zeros(2,t);

for k = 1:t
    measurement_noise = mvnrnd(measurement_noise_mean,R)';
    measurements(:,k) = C * trueTarget(:,k) + H * measurement_noise;
end


[estimated_states,estimated_covariances] = batch_KF(measurements,A,B,C,H,Q,R,t,x0_bar,P0);

figure;
plot(trueTarget(1,:),trueTarget(2,:),LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319");
title("True Target Trajectory vs. Estimated Trajectory");
ylabel("y position");
xlabel("x position");
plot(measurements(1,:),measurements(2,:),"r.");

legend("True Target Trajectory","Estimated Trajectory","Measurements");
grid on;

NEESs = zeros(1,t);

for k = 1:length(NEESs)
    NEESs(k) = (trueTarget(:,k) - estimated_states(:,k))' * inv(estimated_covariances{k}) * (trueTarget(:,k) - estimated_states(:,k));
end

figure;
plot(1:t,NEESs);
grid on;
hold on;
yline(chi2inv(0.005,4));
yline(chi2inv(1-0.005,4));
title("NEES");

Nmc = 100;
NEESmc = zeros(Nmc,t);
for i = 1:Nmc
    trueTarget = zeros(4,t);
    trueTarget(:,1) = mvnrnd(x0_bar,P0)';

    for k = 2:t
        process_noise = mvnrnd(process_noise_mean,Q)';
        trueTarget(:,k) = A * trueTarget(:,k-1) + B * process_noise;
    end


    measurements = zeros(2,t);
    
    for k = 1:t
        measurement_noise = mvnrnd(measurement_noise_mean,R)';
        measurements(:,k) = C * trueTarget(:,k) + H * measurement_noise;
    end
    
    
    [estimated_states,estimated_covariances] = batch_KF(measurements,A,B,C,H,Q,R,t,x0_bar,P0);
        


    for k = 1:t
        NEESmc(i,k) = (trueTarget(:,k) - estimated_states(:,k))' * inv(estimated_covariances{k}) * (trueTarget(:,k) - estimated_states(:,k));
    end

end

ANEES = 1/Nmc*sum(NEESmc);

gamma_min = chi2inv(0.005, Nmc*4);
gamma_max = chi2inv(1 - 0.005, Nmc*4);
lower_threshold = gamma_min/Nmc;
upper_threshold = gamma_max/Nmc;

figure;
plot(1:t,ANEES);
grid on;
hold on;
yline(lower_threshold);
yline(upper_threshold);
title("ANEES");

