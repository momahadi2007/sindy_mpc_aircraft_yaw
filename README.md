# sindy_mpc_aircraft_yaw
# **Read Me: Circular Trajectory Tracking using PID, MPC, and SINDy-MPC**

## **Description**

This MATLAB script implements a simulation of a system tracking a circular trajectory using three control strategies: PID, Model Predictive Control (MPC), and Sparse Identification of Nonlinear Dynamics (SINDy)-based MPC. It compares the tracking performance of these controllers in terms of pitch angle and control input.

---

## **Key Components**

### **1. System Definition**
- The system is modeled as a 2nd-order state-space system:
  - \( A \): State matrix.
  - \( B \): Input matrix.
  - \( C \): Output matrix.
  - \( D \): Feedthrough matrix.
- **Moment of Inertia**: \( I_{yy} = 0.07 \) (kg·m²).

---

### **2. Discretization**
- The continuous-time system is discretized using a sampling time of \( T_s = 0.01 \) seconds.
- Discretized matrices (\( A_d \), \( B_d \)) are used for control algorithms.

---

### **3. Desired Trajectory**
- A **circular trajectory** is designed:
  - **Radius**: \( r = 10 \).
  - **Angular velocity**: \( \omega = \frac{2\pi}{T} \), where \( T = 20 \) seconds.
  - Reference pitch (\( \theta_{\text{ref}} \)) and yaw (\( \psi_{\text{ref}} \)) angles are calculated.

---

### **4. Controllers**
#### **PID Controller**
- Gains:
  - \( K_p = 3 \),
  - \( K_i = 0.01 \),
  - \( K_d = 0.5 \).
- Uses error, integral of error, and derivative of error to compute the control input.

#### **MPC Controller**
- Prediction Horizon (\( N_p \)): 20.
- Control Horizon (\( N_c \)): 10.
- Cost function minimizes:
  - State errors (\( Q \)): \( Q = \begin{bmatrix} 10 & 0 \\ 0 & 1 \end{bmatrix} \),
  - Input effort (\( R = 0.1 \)).
- Solved using `fmincon`.

#### **SINDy-MPC Controller**
- Sparse Identification of Nonlinear Dynamics (SINDy) approximates system dynamics using a library of polynomial terms up to 2nd order.
- Cost function similar to MPC but uses SINDy-identified dynamics.

---

### **5. Training Data for SINDy**
- Generated using random control inputs to simulate system behavior.
- Sparse regression (`sparsifyDynamics`) identifies a sparse coefficient matrix \( \Xi \).

---

### **6. Simulation**
- Simulates all three controllers over the trajectory and records:
  - Pitch angle tracking (\( \theta(t) \)),
  - Control input (\( u(t) \)).

---

## **Outputs**

1. **Plots:**
   - **Pitch Angle Tracking**: Compares reference trajectory and actual trajectories from PID, MPC, and SINDy-MPC.
   - **Control Input**: Displays control effort for each controller.
   - **Trajectory in Pitch-Yaw Plane**: Visualizes circular trajectory tracking.

---

## **How to Run**
1. Ensure MATLAB and its Optimization Toolbox are installed.
2. Run the script in MATLAB.
3. The results will be displayed as plots.

---

## **Key Functions**
1. `sparsifyDynamics`: Performs sparse regression to identify dynamics.
2. `constructLibrary`: Builds the polynomial library for SINDy.
3. `compute_mpc_cost`: Evaluates the cost function for MPC.
4. `compute_sindy_mpc_cost`: Evaluates the cost function for SINDy-MPC.

---

## **Parameters**
- **Sampling Time**: \( T_s = 0.01 \) s.
- **Control Input Limits**: \( u \in [-15^\circ, 15^\circ] \).
- **Prediction Horizon**: \( N_p = 20 \).
- **Control Horizon**: \( N_c = 10 \).
- **SINDy Regularization**: \( \lambda = 0.1 \).

---

## **Dependencies**
- MATLAB Optimization Toolbox (`fmincon`).

---

## **Contact**
For questions or support, contact the author of the code.
