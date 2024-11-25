clear; clc;

%% System Parameters
Iyy = 0.07; % Moment of inertia about y-axis (kg.m^2)
A = [0 1; 0 0]; % State matrix
B = [0; 1/Iyy]; % Input matrix
C = [1 0]; % Output matrix (only pitch angle is measured)
D = 0;

% State-space system
sys = ss(A, B, C, D);

%% Discretize System
Ts = 0.01; % Sampling time (s)
sys_d = c2d(sys, Ts);
[Ad, Bd, ~, ~] = ssdata(sys_d);

%% Design Circular Trajectory
T = 20; % Total time (s)
time = 0:Ts:T; % Time vector
radius = 10; % Circle radius
omega = 2 * pi / T; % Angular velocity

% Desired trajectory (circle)
theta_ref_traj = radius * sin(omega * time); % Pitch angle
yaw_ref_traj = radius * cos(omega * time); % Yaw angle (for visualization)

%% PID Parameters
Kp = 3; Ki = 0.01; Kd = 0.5; % Gains
integral_error = 0; prev_error = 0;

%% MPC Parameters
Np = 20; % Prediction horizon
Nc = 10; % Control horizon
Q = [10 0; 0 1]; % State weighting matrix
R = 0.1; % Input weighting matrix
u_min = -15 * pi/180; % Control input limits (radians)
u_max = 15 * pi/180;

%% Initial Conditions
x_pid = [0; 0]; % Initial state for PID
x_mpc = [0; 0]; % Initial state for MPC
x_sindy_mpc = [0; 0]; % Initial state for SINDy-MPC

theta_traj_pid = zeros(length(time), 1);
theta_traj_mpc = zeros(length(time), 1);
theta_traj_sindy_mpc = zeros(length(time), 1);

u_traj_pid = zeros(length(time), 1);
u_traj_mpc = zeros(length(time), 1);
u_traj_sindy_mpc = zeros(length(time), 1);

%% Generate Training Data for SINDy
N_train = 1000;
x_train = [0; 0];
u_train = (rand(N_train, 1) - 0.5) * 10 * pi / 180; % Random control inputs
X_train = zeros(N_train, 2);
dXdt_train = zeros(N_train, 2);

for k = 1:N_train
    X_train(k, :) = x_train';
    x_next = Ad * x_train + Bd * u_train(k);
    dXdt_train(k, :) = (x_next - x_train)' / Ts;
    x_train = x_next;
end

% Construct Theta for SINDy
poly_order = 2; % Polynomial order
Theta_train = [];
for i = 0:poly_order
    for j = 0:poly_order - i
        Theta_train = [Theta_train, (X_train(:, 1).^i) .* (X_train(:, 2).^j)];
    end
end
Theta_train = [Theta_train, u_train];

% SINDy Sparse Regression
lambda = 0.1;
Xi = sparsifyDynamics(Theta_train, dXdt_train, lambda, size(X_train, 2));

%% Simulation Loop
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

for k = 1:length(time)
    theta_ref = theta_ref_traj(k);
    
    % ---------------- PID Control ----------------
    error = theta_ref - x_pid(1);
    integral_error = integral_error + error * Ts;
    derivative_error = (error - prev_error) / Ts;
    u_pid = Kp * error + Ki * integral_error + Kd * derivative_error;
    u_pid = max(min(u_pid, u_max), u_min);
    prev_error = error;
    x_pid = Ad * x_pid + Bd * u_pid;
    
    theta_traj_pid(k) = x_pid(1);
    u_traj_pid(k) = u_pid;
    
    % ---------------- MPC Control ----------------
    cost_func_mpc = @(u_seq) compute_mpc_cost(Ad, Bd, Q, R, x_mpc, theta_ref, u_seq, Np);
    u_seq_mpc = fmincon(cost_func_mpc, zeros(Nc, 1), [], [], [], [], ...
                        u_min * ones(Nc, 1), u_max * ones(Nc, 1), [], options);
    u_mpc = u_seq_mpc(1);
    x_mpc = Ad * x_mpc + Bd * u_mpc;
    
    theta_traj_mpc(k) = x_mpc(1);
    u_traj_mpc(k) = u_mpc;
    
    % ---------------- SINDy-MPC Control ----------------
    cost_func_sindy_mpc = @(u_seq) compute_sindy_mpc_cost(x_sindy_mpc, theta_ref, u_seq, Xi, Np, poly_order, Q, R, Ts);
    u_seq_sindy = fmincon(cost_func_sindy_mpc, zeros(Nc, 1), [], [], [], [], ...
                          u_min * ones(Nc, 1), u_max * ones(Nc, 1), [], options);
    u_sindy = u_seq_sindy(1);
    
    % Simulate with SINDy dynamics
    Theta_k = constructLibrary(x_sindy_mpc, u_sindy, poly_order);
    dxdt_sindy = Theta_k * Xi;
    x_sindy_mpc = x_sindy_mpc + Ts * dxdt_sindy';
    
    theta_traj_sindy_mpc(k) = x_sindy_mpc(1);
    u_traj_sindy_mpc(k) = u_sindy;
end

%% Plot Results
figure;
subplot(3, 1, 1);
plot(time, theta_ref_traj * 180/pi, 'k--', 'LineWidth', 1.5); hold on;
plot(time, theta_traj_pid * 180/pi, 'r', 'LineWidth', 1.5);
plot(time, theta_traj_mpc * 180/pi, 'b', 'LineWidth', 1.5);
plot(time, theta_traj_sindy_mpc * 180/pi, 'g', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Yaw Angle (deg)');
legend('Reference', 'PID', 'MPC', 'SINDy-MPC');
title('Pitch Angle Tracking');
grid on;

subplot(3, 1, 2);
plot(time, u_traj_pid * 180/pi, 'r', 'LineWidth', 1.5); hold on;
plot(time, u_traj_mpc * 180/pi, 'b', 'LineWidth', 1.5);
plot(time, u_traj_sindy_mpc * 180/pi, 'g', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Control Input (deg)');
legend('PID', 'MPC', 'SINDy-MPC');
title('Control Input');
grid on;

subplot(3, 1, 3);
plot(theta_ref_traj* 180/pi, yaw_ref_traj, 'k--', 'LineWidth', 1.5); hold on;
plot(theta_traj_pid * 180/pi, yaw_ref_traj, 'r', 'LineWidth', 1.5);
plot(theta_traj_mpc * 180/pi, yaw_ref_traj, 'b', 'LineWidth', 1.5);
plot(theta_traj_sindy_mpc * 180/pi, yaw_ref_traj, 'g', 'LineWidth', 1.5);
xlabel('Pitch Angle (deg)');
ylabel('Yaw Angle (deg)');
legend('Reference', 'PID', 'MPC', 'SINDy-MPC');
title('Trajectory in Pitch-Yaw Plane');
grid on;

%% Functions
function Xi = sparsifyDynamics(Theta, dXdt, lambda, n)
    Xi = Theta \ dXdt;
    for k = 1:10
        small_idx = abs(Xi) < lambda;
        Xi(small_idx) = 0;
        for i = 1:n
            big_idx = ~small_idx(:, i);
            if any(big_idx)
                Xi(big_idx, i) = Theta(:, big_idx) \ dXdt(:, i);
            end
        end
    end
end

function Theta = constructLibrary(x, u, poly_order)
    Theta = [];
    for i = 0:poly_order
        for j = 0:poly_order - i
            Theta = [Theta, (x(1)^i) * (x(2)^j)];
        end
    end
    Theta = [Theta, u];
end

function J = compute_mpc_cost(Ad, Bd, Q, R, x0, theta_ref, u_seq, Np)
    x = x0;
    J = 0;
    for i = 1:Np
        x = Ad * x + Bd * u_seq(min(i, length(u_seq)));
        theta_error = theta_ref - x(1);
        J = J + theta_error' * Q(1, 1) * theta_error + u_seq(min(i, length(u_seq)))' * R * u_seq(min(i, length(u_seq)));
    end
end

function J = compute_sindy_mpc_cost(x0, theta_ref, u_seq, Xi, Np, poly_order, Q, R, Ts)
    x = x0;
    J = 0;
    for i = 1:Np
        Theta_k = constructLibrary(x, u_seq(min(i, length(u_seq))), poly_order);
        dxdt_sindy = Theta_k * Xi;
        x = x + Ts * dxdt_sindy';
        theta_error = theta_ref - x(1);
        J = J + theta_error' * Q(1, 1) * theta_error + u_seq(min(i, length(u_seq)))' * R * u_seq(min(i, length(u_seq)));
    end
end
