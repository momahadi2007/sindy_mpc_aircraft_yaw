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

%% Define Trajectory
% Trajectory parameters
t_takeoff = 5; % Time for takeoff (seconds)
t_cruise = 10; % Time for cruise (seconds)
t_landing = 5; % Time for landing (seconds)
z_max = 40; % Maximum altitude (meters)
x_cruise = 100; % Cruise distance (meters)

% Generate trajectory
[trajectory, t_vec] = design_trajectory(Ts, t_takeoff, t_cruise, t_landing, z_max, x_cruise)


%% PID Parameters
Kp = 3;    % Proportional gain
Ki = 0.01; % Integral gain
Kd = 0.5;  % Derivative gain
integral_error = 0; % Initialize integral error for PID
prev_error = 0;     % Initialize previous error for PID

%% MPC Parameters
Np = 20; % Prediction horizon
Nc = 10; % Control horizon
Q = eye(2) * 10; % State weighting matrix
R = 0.1; % Input weighting matrix
u_min = -15 * pi/180; % Elevator deflection limits (radians)
u_max = 15 * pi/180;

% Precompute MPC cost matrices
[H, F, Phi, Gamma] = compute_mpc_matrices(Ad, Bd, Q, R, Np, Nc);

%% Generate Training Data for SINDy
N_train = 1000; % Training data size
[X_train, u_train, Theta_train, dXdt_train] = generate_sindy_training_data(Ad, Bd, Ts, N_train, u_min, u_max);

% Perform SINDy Sparse Regression
lambda = 0.1; % Sparsification threshold
Xi = sparsifyDynamics(Theta_train, dXdt_train, lambda, size(X_train, 2));

%% Simulation Setup
N_sim = length(t_vec);
x_pid = [0; 0]; x_mpc = [0; 0]; x_sindy_mpc = [0; 0];
theta_traj_pid = zeros(N_sim, 1);
theta_traj_mpc = zeros(N_sim, 1);
theta_traj_sindy_mpc = zeros(N_sim, 1);
u_traj_pid = zeros(N_sim, 1);
u_traj_mpc = zeros(N_sim, 1);
u_traj_sindy_mpc = zeros(N_sim, 1);

%% Simulation Loop
options_mpc = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
for k = 1:N_sim
    % Desired trajectory
    theta_ref = trajectory.z(k);
    
    % ------------------ PID Control ------------------
    error = theta_ref - x_pid(1);
    integral_error = integral_error + error * Ts;
    derivative_error = (error - prev_error) / Ts;
    u_applied_pid = Kp * error + Ki * integral_error + Kd * derivative_error;
    u_applied_pid = max(min(u_applied_pid, u_max), u_min); % Saturation
    x_pid = Ad * x_pid + Bd * u_applied_pid;
    prev_error = error;

    % Log PID results
    theta_traj_pid(k) = x_pid(1);
    u_traj_pid(k) = u_applied_pid;

    % ------------------ MPC Control ------------------
    x_pred_error = repmat(theta_ref, Np, 1) - Phi * x_mpc;
    cost_func_mpc = @(u_seq) 0.5 * u_seq' * H * u_seq + x_pred_error' * F * u_seq;
    u_seq_mpc = fmincon(cost_func_mpc, zeros(Nc, 1), [], [], [], [], ...
        u_min * ones(Nc, 1), u_max * ones(Nc, 1), [], options_mpc);
    u_mpc = u_seq_mpc(1);
    x_mpc = Ad * x_mpc + Bd * u_mpc;

    % Log MPC results
    theta_traj_mpc(k) = x_mpc(1);
    u_traj_mpc(k) = u_mpc;

    % ------------------ SINDy-MPC Control ------------------
    Theta_sindy = constructLibrary(x_sindy_mpc, u_traj_sindy_mpc(max(k-1,1)), 2);
    x_sindy_next = Theta_sindy * Xi';
    x_pred_error_sindy = repmat(theta_ref, Np, 1) - x_sindy_next(1:Np);
    cost_func_sindy = @(u_seq) 0.5 * u_seq' * H * u_seq + x_pred_error_sindy' * F * u_seq;
    u_seq_sindy = fmincon(cost_func_sindy, zeros(Nc, 1), [], [], [], [], ...
        u_min * ones(Nc, 1), u_max * ones(Nc, 1), [], options_mpc);
    u_sindy = u_seq_sindy(1);
    x_sindy_mpc = Ad * x_sindy_mpc + Bd * u_sindy;

    % Log SINDy-MPC results
    theta_traj_sindy_mpc(k) = x_sindy_mpc(1);
    u_traj_sindy_mpc(k) = u_sindy;
end

%% Plot Results
plot_results(t_vec, trajectory, theta_traj_pid, u_traj_pid, theta_traj_mpc, u_traj_mpc, ...
    theta_traj_sindy_mpc, u_traj_sindy_mpc);
%% Helper Functions
function [trajectory, t_vec] = design_trajectory(Ts, t_takeoff, t_cruise, t_landing, z_max, x_cruise)

    % Time segments
    t1 = 0:Ts:t_takeoff-Ts; % Takeoff phase
    t2 = t1(end) + Ts:Ts:t1(end) + t_cruise; % Cruise phase
    t3 = t2(end) + Ts:Ts:t2(end) + t_landing; % Landing phase

    % Total time vector
    t_vec = [t1, t2, t3];

    % Trajectory segments
    z_takeoff = linspace(0, z_max, length(t1)); % Vertical climb
    z_cruise = z_max * ones(1, length(t2)); % Maintain altitude
    z_landing = linspace(z_max, 0, length(t3)); % Descent

    x_takeoff = zeros(1, length(t1)); % No horizontal movement
    x_cruise = linspace(0, x_cruise, length(t2)); % Horizontal motion
    x_landing = x_cruise(end) * ones(1, length(t3)); % Maintain horizontal position

    % Combine all segments
    trajectory.x = [x_takeoff, x_cruise, x_landing];
    trajectory.z = [z_takeoff, z_cruise, z_landing];

end

function plot_results(z_desired, z_traj_pid, z_traj_mpc, z_traj_sindy_mpc, u_traj_pid, u_traj_mpc, u_traj_sindy_mpc, Ts)
    % Plot the results of the mission profile
    
    % Time vector
    N_sim = length(z_desired);
    time = (0:N_sim-1) * Ts;
    
    % Plot Position (Altitude)
    figure;
    subplot(2, 1, 1);
    plot(time, z_desired, 'k--', 'LineWidth', 1.5); hold on;
    plot(time, z_traj_pid, 'r', 'LineWidth', 1.5);
    plot(time, z_traj_mpc, 'b', 'LineWidth', 1.5);
    plot(time, z_traj_sindy_mpc, 'g', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Altitude (m)');
    title('Altitude Tracking');
    legend('Desired', 'PID', 'MPC', 'SINDy-MPC');
    hold off;

    % Plot Control Inputs
    subplot(2, 1, 2);
    plot(time, u_traj_pid, 'r', 'LineWidth', 1.5); hold on;
    plot(time, u_traj_mpc, 'b', 'LineWidth', 1.5);
    plot(time, u_traj_sindy_mpc, 'g', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Control Input');
    title('Control Effort');
    legend('PID', 'MPC', 'SINDy-MPC');
    hold off;
end
function [X_train, u_train, Theta_train, dXdt_train] = generate_sindy_training_data(Ad, Bd, Ts, N_train, u_min, u_max, poly_order)
    % Generate training data for SINDy
    
    % Initialize variables
    n_states = size(Ad, 1); % Number of states
    n_inputs = size(Bd, 2); % Number of inputs
    X_train = zeros(N_train, n_states); % States
    u_train = zeros(N_train, n_inputs); % Inputs
    dXdt_train = zeros(N_train, n_states); % State derivatives
    Theta_train = []; % Candidate functions
    
    % Initial state
    x = zeros(n_states, 1);
    
    for k = 1:N_train
        % Random control input within limits
        u = (u_max - u_min) .* rand(n_inputs, 1) + u_min;
        u_train(k, :) = u';
        
        % Record state and compute derivative
        X_train(k, :) = x';
        x_next = Ad * x + Bd * u; % Next state
        dXdt_train(k, :) = ((x_next - x) / Ts)'; % Finite difference
        
        % Generate candidate functions for SINDy
        Theta_k = construct_library(x, u, poly_order);
        Theta_train = [Theta_train; Theta_k];
        
        % Update state
        x = x_next;
    end
end

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
function [H, F, Phi, Gamma] = compute_mpc_matrices(Ad, Bd, Q, R, Np, Nc)
    % Compute MPC Matrices for a given system and horizons
    %
    % Inputs:
    % Ad, Bd - Discrete state-space matrices
    % Q, R   - State and input weight matrices
    % Np     - Prediction horizon
    % Nc     - Control horizon
    %
    % Outputs:
    % H, F   - Quadratic cost matrices
    % Phi    - State transition matrix
    % Gamma  - Input effect matrix

    % State and input dimensions
    nx = size(Ad, 1); % Number of states
    nu = size(Bd, 2); % Number of inputs

    % Initialize Phi and Gamma
    Phi = zeros(nx * Np, nx);
    Gamma = zeros(nx * Np, nu * Nc);

    % Build Phi and Gamma matrices
    for i = 1:Np
        Phi((i-1)*nx+1:i*nx, :) = Ad^i;
        for j = 1:min(i, Nc)
            Gamma((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = Ad^(i-j) * Bd;
        end
    end

    % Build block-diagonal Q_bar and R_bar matrices
    Q_bar = kron(eye(Np), Q);
    R_bar = kron(eye(Nc), R);

    % Compute cost matrices H and F
    H = Gamma' * Q_bar * Gamma + R_bar;
    F = Gamma' * Q_bar * Phi;
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
