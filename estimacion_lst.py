# -*- coding: utf-8 -*-
"""

@author: gsanchez, ntrivisonno
"""

import numpy as np
from scipy import linalg
import casadi as cs
import h5py

from parameters import parameters
from fluid_prop import fluid_prop
from plot_data import plot_data
from plot_data_time import plot_data_time
from save_data import save_data
from plot_data_noise import plot_data_noise

import casadi_model_equations as model

import matplotlib.pyplot as plt

m, diam, xcg, ycg, zcg, Ixx, Iyy, Izz, steps, dt = parameters('./Data/data_F01.dat')

S = np.pi * (0.5 * diam) ** 2
g = 9.81  # gravity

# Path to data
Resul = ['Resu_RBD/Caso_F01/']
# Resul = ['Resu_RBD/Caso_F01_unificated/']
# Resul = ['/home/ntrivi/Documents/Tesis/RBD/Resu_ref/Wernert_AIAA2010_7460/Caso_F01_unificated/']

# Read forces-moments data
data = np.loadtxt(Resul[0] + 'Forces_proc.txt', delimiter=',', skiprows=1)
datam = np.loadtxt(Resul[0] + 'Moments_proc.txt', delimiter=',', skiprows=1)

Ndata = 10000  # data.shape[0]

if Ndata < data.shape[0]:
    data = data[:Ndata]
    datam = datam[:Ndata]

# Read data from hdf5 and put in xned list
xned = []
h5f = h5py.File(Resul[0] + 'Data.hdf5', 'r')
xned.append(h5f['/Inertial_coord'][1:Ndata + 1, :])
h5f.close()

# Propiedades fluido vs altura
rho, mu, c = fluid_prop(xned[0][:, 2], 0)

# print(data.shape)

# Encabezado del txt:
# Time, alpha, beta, V_inf (= V_t), u(v_body_X), v(v_body_Y), w(v_body_Z), p, q, r, gx, gy, gz, FX, FY, FZ

time = data[:, 0]
alpha = data[:, 1]
beta = data[:, 2]
delta2 = ((np.sin(beta)) ** 2 + (np.cos(beta)) ** 2 * (np.sin(alpha)) ** 2)  # alpha2
vt = data[:, 3]
mach = vt / c
u = data[:, 4]  # vel_body_X
v = data[:, 5]  # vel_body_Y
w = data[:, 6]  # vel_body_Z
p = data[:, 7]
q = data[:, 8]
r = data[:, 9]
grav = data[:, 10:13]  # gx, gy, gz
F_body = data[:, 13:16]  # FX, FY, FZ
F_body = np.column_stack([F_body, datam[:, 6:9]])  # Add moments to F_body

""" Arranco la configuración de MHE """

Nx = 8  # cant de coef a estimar
Ny = 6
Nw = Nx
Nv = Ny
Np = 8  # Cantidad de parametros al solver
Nt = 5  # Longitud del horizonte de estimación
Nu = 0

Q = np.diag([.1] * Nx)  # matriz de covarianza de ruido de proceso
R = np.diag([1] * Ny)  # matriz de covarianza de ruido de  medición
P = np.diag([10.] * Nx)  # matriz de covarianza de estimación inicial

Q_inv = linalg.inv(Q)
R_inv = linalg.inv(R)
P_inv = linalg.inv(P)

# Declare model variables
x_casadi = cs.SX.sym('x', Nx)
# w = cs.SX.sym('w', Nw)
v_casadi = cs.SX.sym('v', Nv)
p_casadi = cs.SX.sym('p', Np)

# The model is defined as follows
'''
    x[0]: Cd0
    x[1]: Cl_alpha
    x[2]: Cd2
    x[3]: Cn_p_alfa
    x[4]: Clp
    x[5]: Cm_alpha
    x[6]: Cm_p_alpha
    x[7]: Cm_q
    p[0]: vt
    p[1]: alpha
    p[2]: beta
    p[3]: delta2
    p[4]: p(rolling)
    p[5]: q
    p[6]: r
    p[7]: rho
    y[0]: Fx
    y[1]: Fy
    y[2]: Fz
    y[3]: Mx
    y[4]: My
    y[5]: Mz
'''

# Model equations (there is none)
# f_rhs = []
# f_rhs = cs.vertcat(*f_rhs)

h_rhs = model.meas(x_casadi, p_casadi, S, diam)
h_rhs = cs.vertcat(*h_rhs)

# Continuous time dynamics
# f = cs.Function('f', [x, w], [f_rhs])
h = cs.Function('h', [x_casadi, p_casadi], [h_rhs])

"""
Here we start the MHE formulation with an empty NLP
"""

N = 10000

optimization_variables = []
optimization_parameters = []
optimization_initial_condition = []

optimization_variables_lb = []
optimization_variables_ub = []

x_lb = -cs.DM.inf(Nx)
x_ub = cs.DM.inf(Nx)

# x_lb[7] = -17.0
# x_ub[7] = -4.0
# x_lb = cs.DM([0.0,    0.0,    0.0,     -cs.inf,    -cs.inf,     -cs.inf,   -cs.inf, -20.0])
# x_ub = cs.DM([cs.inf, cs.inf, cs.inf,   0.0,       0.0,         0.0 ,       cs.inf,  0.0])


# w_lb = -cs.DM.inf(nw)
# w_ub = cs.DM.inf(nw)= _x[0] + _x[2] * _p[3
v_lb = -cs.DM.inf(Nv)
v_ub = cs.DM.inf(Nv)
# v_lb = -cs.DM.zeros(Nv)
# v_ub = cs.DM.zeros(Nv)

J = 0

# state_constraints = []
# state_constraints_lb = []
# state_constraints_ub = []

measurement_constraints = []
measurement_constraints_lb = []
measurement_constraints_ub = []

# Our first parameter, x0bar
x0bar = cs.SX.sym('x_0_bar', Nx)  # The arrival cost is a parameter

# We can set these as parameters later
P_x = cs.DM.eye(Nx)  # Arrival cost weighting matrix
P_x[2, 2] = 1E6
# Q_mhe = cs.DM.eye(Nx)
R_mhe = 10 * cs.DM.eye(Ny)
R_mhe[5, 5] = 0.10000
R_mhe[4, 4] = 0.10000
R_mhe[3, 3] = 0.10000

# Quick and easy... but don't know if it can be done in Matlab

# Esto crea una lista de N elementos y en cada posicion de la lista,
# el MX con el nombre del intervalo. Por ejemplo:
# Xk = [MX(X_0), MX(X_1), ..., MX(X_N-1)]
Xk = [cs.SX.sym('X_' + str(i), Nx) for i in range(N)]
Yk = [cs.SX.sym('Y_' + str(i), Ny) for i in range(N)]
Vk = [cs.SX.sym('V_' + str(i), Nv) for i in range(N)]
Pk = [cs.SX.sym('P_' + str(i), Np) for i in range(N)]
# Wk = [cs.SX.sym('W_' + str(i), Nw) for i in range(N-1)]

# Next the state and measurement constraints
for i in range(N-1):
    # We calculate X(i+1) with RK4
    # Fk = F_RK4(Xk[i], Wk[i])

    # state_constraints += [Xk[i+1] - Fk]
    # Add equality constraints
    # state_constraints_lb += [cs.DM.zeros(nx)]
    # state_constraints_ub += [cs.DM.zeros(nx)]

    measurement_constraints += [Yk[i] - h(Xk[i], Pk[i]) - Vk[i]]
    # Add equality constraints
    measurement_constraints_lb += [cs.DM.zeros(Ny)]
    measurement_constraints_ub += [cs.DM.zeros(Ny)]

    # First the arrival cost
    if i == 0:
        J += cs.mtimes([(Xk[0] - x0bar).T, P_x, (Xk[0] - x0bar)])
        optimization_parameters += [x0bar]

    # Stage cost for the process noise
    # J += cs.mtimes([Wk[i].T, Q_mhe, Wk[i]])
    # Stage cost for the measurement noise
    J += cs.mtimes([Vk[i].T, R_mhe, Vk[i]])

    # optimization_variables += [Xk[i], Wk[i], Vk[i]]
    # optimization_variables_lb += [x_lb, w_lb, v_lb]
    # optimization_variables_ub += [x_ub, w_ub, v_ub]
    # optimization_variables += [Xk[i], Vk[i]]
    # optimization_variables_lb += [x_lb, v_lb]
    # optimization_variables_ub += [x_ub, v_ub]
    # optimization_parameters += [Yk[i], Pk[i]]


# Measurement constraint for the last measurement interval
# Xk = cs.MX.sym('X_' + str(N-1), nx)
# Yk = cs.MX.sym('Y_' + str(N-1), ny)
# Vk = cs.MX.sym('V_' + str(N-1), nv)
measurement_constraints += [Yk[N-1] - h(Xk[N-1], Pk[N-1]) - Vk[N-1]]

# stage cost for the last measurement interval
J += cs.mtimes([Vk[N-1].T, R_mhe, Vk[N-1]])

# optimization_variables += [Xk[N-1], Vk[N-1]]
# optimization_variables_lb += [x_lb, v_lb]
# optimization_variables_ub += [x_ub, v_ub]
# optimization_parameters += [Yk[N-1], Pk[N-1]]


for i in range(N):
    optimization_variables += [Xk[i]]
    optimization_variables_lb += [x_lb]
    optimization_variables_ub += [x_ub]

for i in range(N):
    optimization_variables += [Vk[i]]
    optimization_variables_lb += [v_lb]
    optimization_variables_ub += [v_ub]

for i in range(N):
    optimization_parameters += [Yk[i]]
for i in range(N):
    optimization_parameters += [Pk[i]]

# nlp_constraints = state_constraints + measurement_constraints
nlp_constraints = measurement_constraints

# Create an NLP solver
prob = {'f': J,
        'x': cs.vertcat(*optimization_variables),
        'g': cs.vertcat(*nlp_constraints),
        'p': cs.vertcat(*optimization_parameters)}
solver = cs.nlpsol('solver', 'ipopt', prob);

# Solve the NLP
optimization_variables_lb = cs.vertcat(*optimization_variables_lb)
optimization_variables_ub = cs.vertcat(*optimization_variables_ub)
optimization_initial_condition = cs.DM.zeros(optimization_variables_lb.shape)

tmin = 0
tmax = N
p_coefs = np.vstack((vt[tmin:tmax], alpha[tmin:tmax], beta[tmin:tmax], delta2[tmin:tmax],
                     p[tmin:tmax], q[tmin:tmax], r[tmin:tmax], rho[tmin:tmax])).T

y_meas = F_body[tmin:tmax, :]

x0 = cs.DM.zeros((Nx,1))

optimization_initial_condition[0] = 0.3
optimization_initial_condition[2] = 3.5

sol = solver(x0=optimization_initial_condition, lbx=optimization_variables_lb, ubx=optimization_variables_ub, lbg=0, ubg=0, p=cs.vertcat(x0, *y_meas, *p_coefs))

xhat = sol['x'][:N*Nx]
xhat = np.reshape(xhat, (N, Nx))

#Save estimated coeficients
# save_data(time, mach, alpha, xhat)

#Plot data vs mach
# plot_data(Resul, data, mach, xhat)
#Plot data v time
plot_data_time(Resul, data, mach, xhat)

# data_ref = np.loadtxt(Resul[0]+'Force_coef_proc.txt', delimiter=',', skiprows=1)
# data_ref = data_ref[:Ndata]
#
# Cd0_ref = data_ref[:,5]
# Cdd2_ref = data_ref[:,6]
#
# fig_size = (15,5)
# leg = ['Estim', 'Used','Alpha','Beta','Tot']
# f, ax = plt.subplots(1,6, figsize=fig_size)
#
# ax[0].plot(time, Cd0_ref,label=leg[1])
# ax[0].plot(time, xhat[:,0],label=leg[0])
#
# ax[1].plot(time, Cdd2_ref,label=leg[1])
# ax[1].plot(time, xhat[:,2],label=leg[0])
# # ax[2].plot(time, xhat[:,0],label=leg[0])
#
# plt.legend()
# plt.show()
