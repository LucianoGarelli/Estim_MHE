# -*- coding: utf-8 -*-
"""

@author: gsanchez, ntrivisonno
"""
import casadi as cs
import matplotlib.pyplot as plt

Ts = 0.1
nx = 2
nw = 2
ny = 1
nv = 1
# Declare model constants
k = 0.16
# Declare model variables
x = cs.SX.sym('x', nx)
w = cs.SX.sym('w', nw)
# v = cs.SX.sym('v', nv)

# Model equations
f_rhs = [-2 * (x[0]**2)*k + w[0], k*x[0]**2 + w[1]]
f_rhs = cs.vertcat(*f_rhs)

h_rhs = [x[0] + x[1]]
h_rhs = cs.vertcat(*h_rhs)

# Continuous time dynamics
f = cs.Function('f', [x, w], [f_rhs])
h = cs.Function('h', [x], [h_rhs])

# Dynamics discretization,
# Fixed step RK4 integrator

M = 4 # RK4 steps per interval
for j in range(1, M):
    k1 = f(x, w)
    k2 = f(x + Ts / 2 * k1, w)
    k3 = f(x + Ts / 2 * k2, w)
    k4 = f(x + Ts * k3, w)
    x_rk4 = x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# x_rk4 = cs.vertcat(*x_rk4)

F_RK4 = cs.Function('F', [x, w], [x_rk4])

x0 = cs.DM([4.5, 0.1])

Nsim = 80

x_sim = cs.DM.zeros(nx, Nsim + 1)
x_sim[:, 0] = x0
y_sim = cs.DM.zeros(ny, Nsim)

w_zero = cs.DM.zeros(nw, 1)
v_zero = cs.DM.zeros(nv)
for j in range(Nsim):
    x_sim[:, j + 1] = F_RK4(x_sim[:, j], w_zero)
    y_sim[j] = h(x_sim[:, j])

plt.figure()
plt.plot(x_sim[0, :].toarray().flatten())
plt.plot(x_sim[1, :].toarray().flatten())
plt.show(block=False)


# Start with an empty NLP
N = 10

optimization_variables = []
optimization_parameters = []
optimization_initial_condition = []

optimization_variables_lb = []
optimization_variables_ub = []

x_lb = -cs.DM.inf(nx)
x_ub = cs.DM.inf(nx)
w_lb = -cs.DM.inf(nw)
w_ub = cs.DM.inf(nw)
v_lb = -cs.DM.inf(nv)
v_ub = cs.DM.inf(nv)

J = 0

state_constraints = []
state_constraints_lb = []
state_constraints_ub = []

measurement_constraints = []
measurement_constraints_lb = []
measurement_constraints_ub = []

# Our first parameter, x0bar
x0bar = cs.MX.sym('x_0_bar', nx)  # The arrival cost is a parameter

# We can set these as parameters later
P_x = cs.DM.eye(nx)  # Arrival cost weighting matrix
Q_mhe = cs.DM.eye(nx)
R_mhe = cs.DM.eye(ny)

# Quick and easy... but don't know if it can be done in Matlab

# Esto crea una lista de N elementos y en cada posicion de la lista,
# el MX con el nombre del intervalo. Por ejemplo:
# Xk = [MX(X_0), MX(X_1), ..., MX(X_N-1)]
Xk = [cs.MX.sym('X_' + str(i), nx) for i in range(N)]
Wk = [cs.MX.sym('W_' + str(i), nw) for i in range(N-1)]
Yk = [cs.MX.sym('Y_' + str(i), ny) for i in range(N)]
Vk = [cs.MX.sym('V_' + str(i), nv) for i in range(N)]

# Next the measurement constraints
for i in range(N-1):
    # Xk = cs.MX.sym('X_' + str(i), nx)
    # Xk_next = cs.MX.sym('X_' + str(i + 1), nx)
    # Wk = cs.MX.sym('W_' + str(i), nw)
    # Yk = cs.MX.sym('Y_' + str(i), ny)
    # Vk = cs.MX.sym('V_' + str(i), nv)

    # We calculate X(i+1) with RK4
    Fk = F_RK4(Xk[i], Wk[i])

    state_constraints += [Xk[i+1] - Fk]
    # Add equality constraints
    state_constraints_lb += [cs.DM.zeros(nx)]
    state_constraints_ub += [cs.DM.zeros(nx)]

    measurement_constraints += [Yk[i] - h(Xk[i]) - Vk[i]]
    # Add equality constraints
    measurement_constraints_lb += [cs.DM.zeros(ny)]
    measurement_constraints_ub += [cs.DM.zeros(ny)]

    # First the arrival cost
    if i == 0:
        J += cs.mtimes([(Xk[0] - x0bar).T, P_x, (Xk[0] - x0bar)])
        optimization_parameters += [x0bar]

    # Stage cost for the process noise
    J += cs.mtimes([Wk[i].T, Q_mhe, Wk[i]])
    # Stage cost for the measurement noise
    J += cs.mtimes([Vk[i].T, R_mhe, Vk[i]])

    optimization_variables += [Xk[i], Wk[i], Vk[i]]
    optimization_variables_lb += [x_lb, w_lb, v_lb]
    optimization_variables_ub += [x_ub, w_ub, v_ub]
    optimization_parameters += [Yk[i]]


# Measurement constraint for the last measurement interval
# Xk = cs.MX.sym('X_' + str(N-1), nx)
# Yk = cs.MX.sym('Y_' + str(N-1), ny)
# Vk = cs.MX.sym('V_' + str(N-1), nv)
measurement_constraints += [Yk[N-1] - h(Xk[N-1]) - Vk[N-1]]

# stage cost for the last measurement interval
J += cs.mtimes([Vk[N-1].T, R_mhe, Vk[N-1]])

optimization_variables += [Xk[N-1], Vk[N-1]]
optimization_variables_lb += [x_lb, v_lb]
optimization_variables_ub += [x_ub, v_ub]
optimization_parameters += [Yk[N-1]]

nlp_constraints = state_constraints + measurement_constraints


# Create an NLP solver
prob = {'f': J,
        'x': cs.vertcat(*optimization_variables),
        'g': cs.vertcat(*nlp_constraints),
        'p': cs.vertcat(*optimization_parameters)}
solver = cs.nlpsol('solver', 'ipopt', prob);

# Solve the NLP
optimization_variables_lb = cs.vertcat(*optimization_variables_lb)
optimization_variables_ub = cs.vertcat(*optimization_variables_ub)

# optimization_variables_lb = cs.vertcat(*optimization_variables_lb)
# optimization_variables_lb = cs.vertcat(*optimization_variables_lb)


sol = solver(x0=optimization_initial_condition, lbx=optimization_variables_lb, ubx=optimization_variables_ub, lbg=0, ubg=0, p=cs.vertcat(x0, y_sim[:N].T))

x_estimated = cs.DM.zeros(nx, Nsim)
x_estimated[0,:N] = sol['x'][0::nx+nv+nw]
x_estimated[1,:N] = sol['x'][1::nx+nv+nw]
current_x0bar = x_estimated[:, 1]
current_measurements = y_sim[:N]
for i in range(N,Nsim):
    current_measurements[0:N-1] = current_measurements[1:N]
    current_measurements[N-1] = y_sim[i]
    current_parameters = cs.vertcat(current_x0bar, current_measurements.T)
    sol = solver(x0=sol['x'], lbx=optimization_variables_lb, ubx=optimization_variables_ub, lbg=0, ubg=0,
                 p=current_parameters)
    # current_x0bar = sol['x'][nx+nw+nv:nx+nw+nv+nx]
    # x_estimated[:, i] = sol['x'][(N-1)*(nx+nw+nv): (N-1)*(nx+nw+nv)+nx]

    x_estimated[0, i] = sol['x'][0::nx + nw + nv][-1]
    x_estimated[1, i] = sol['x'][1::nx + nw + nv][-1]
    current_x0bar = x_estimated[:, i-N+2]


plt.figure()
plt.plot(x_sim[0, :].toarray().flatten(), label='sim_x0')
plt.plot(x_estimated[0, :].toarray().flatten(), label='est_x0')
plt.plot(x_sim[1, :].toarray().flatten(), label='sim_x1')
plt.plot(x_estimated[1, :].toarray().flatten(), label='est_x1')
plt.legend()
plt.show()
