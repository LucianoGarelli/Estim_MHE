# -*- coding: utf-8 -*-
"""

@author: lgenzelis, ntrivisonno
"""

import numpy as np
import matplotlib.pyplot as plt
import mpctools
import casadi
from scipy import linalg
import h5py

from parameters import parameters
from fluid_prop import fluid_prop
from plot_data import plot_data
from plot_data_time import plot_data_time
from save_data import  save_data
from plot_data_noise import plot_data_noise

m, diam, xcg, ycg, zcg, Ixx, Iyy, Izz, steps, dt = parameters('./Data/data_F01.dat')

S = np.pi * (0.5 * diam) ** 2
g= 9.81  # aceleración de la gravedad

# Path to data
Resul = ['Resu_RBD/Caso_F01/']

#Read forces-moments data
data = np.loadtxt(Resul[0]+'Forces_proc_F01_unificated.txt', delimiter=',', skiprows=1)
datam = np.loadtxt(Resul[0]+'Moments_proc_F01_unificated.txt', delimiter=',', skiprows=1)
xned = []
h5f = h5py.File(Resul[0]+'Data.hdf5','r')
# Read data from hdf5
xned.append(h5f['/Inertial_coord'][:])
h5f.close()

# Propiedades fluido vs altura
N = data.shape[0]
#rho, mu, c = fluid_prop(xned[0][1:N+1,2], 0) # cuando se usa datos de data.hdf5

print(data.shape)

# Encabezado del txt:
# Time, alpha, beta, V_inf (= V_t), u(v_body_X), v(v_body_Y), w(v_body_Z), p, q, r, gx, gy, gz, FX, FY, FZ,ZE

time = data[:,0]
alpha = data[:, 1]
beta = data[:, 2]
delta2 = ((np.sin(beta))**2 + (np.cos(beta))**2*(np.sin(alpha))**2) # alpha2
vt = data[:, 3]
u = data[:, 4]  # vel_body_X
v = data[:, 5]  # vel_body_Y
w = data[:, 6]  # vel_body_Z
p = data[:, 7]
q = data[:, 8]
r = data[:, 9]
grav = data[:, 10:13]  # gx, gy, gz
F_body = data[:, 13:16]  # FX, FY, FZ
F_body = np.column_stack([F_body,datam[:, 6:9]]) #Add moments to F_body
ze = data[:,16]
rho, mu, c = fluid_prop(ze, 0)
mach = vt/c
#Add noise
add_noise = False
plot_noise = False
if add_noise:
    n_mean = 0
    n_std = 0.05
    noise = np.random.normal(n_mean, n_std, F_body.shape)
    if plot_noise:
        plot_data_noise(time,F_body,noise)
    F_body = F_body + noise

Ncoef = 8 # cant de coef a estimar
Ny = 6
Nw = Ncoef
Nv = Ny # cant ruido medicion, simil cant mediciones
Np = 8 # cant de parametros al solver
Nt = 1  # horizonte
Nu = 0

Q = np.diag([.1] * Ncoef)  # matrix de covarianza de ruido de proceso
#Q = np.diag([1E1, 1E1, 1E4, 1E-3, 1E1, 1E1, 1E5, 1E5])
Q[0,0] = 10
Q[2,2] = 1
R = np.diag([1]*Ny)     # matrix de covarianza de ruido de  medición
#P = np.diag([10.] * Ncoef)    # matrix de covarianza de estimación inicial
#           [Cd0, Cl_alpha, Cd2, Cn_p_alpha, Clp, Cm_alpha, Cm_alpha, Cm_q]
P = np.diag([1E4, 1E1, 5E5, 1E-3, 1E1, 1E1, 1E5, 1E5]) # matrix de covarianza de estimación inicial

Q_inv = linalg.inv(Q)
R_inv = linalg.inv(R)

def ode(x, u, w):
    dxdt = 0. + w
    return np.array(dxdt)

def meas(x, p):
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
    qdy = 0.5 * p[7] * p[0] ** 2
    y = casadi.SX.zeros(Ny)

    ca = np.cos(p[1])
    sa = np.sin(p[1])
    cb = np.cos(p[2])
    sb = np.sin(p[2])
    Cd = x[0] + x[2] * p[3]
    CL_alfa = x[1]
    Cn_p_alfa = x[3]
    Clp = x[4]
    Cm_alfa = x[5]
    Cm_p_alfa = x[6]
    Cm_q = x[7]
    #Forces
    y[0] = qdy*S*(-Cd*ca*cb + CL_alfa*(sb**2 + sa**2 * cb**2))
    y[1] = qdy*S*(-Cd*sb - CL_alfa*(ca*sb*cb) - Cn_p_alfa*p[4]*diam*(sa*cb)/p[0])
    y[2] = qdy*S*(-Cd*sa*cb - CL_alfa*(sa*ca*cb**2) + Cn_p_alfa*p[4]*diam*sb/p[0])
    #Moments
    y[3] = qdy*S*diam*(p[4]*diam/p[0])*Clp
    y[4] = qdy*S*diam*(Cm_alfa*(sa * cb) - (p[4]*diam/p[0])*(-Cm_p_alfa)*sb + (diam / p[0]) * Cm_q * p[5])
    y[5] = qdy*S*diam*(-Cm_alfa*sb - (p[4]*diam/p[0])*(-Cm_p_alfa)*(sa*cb) + (diam/p[0])*Cm_q * p[6])
    assert Ny == 6
    return y

# no importa el Delta que ponga (por cómo es mi "ode")
F = mpctools.getCasadiFunc(ode, [Ncoef, Nu, Nw], ["x", "u", "w"], "F", rk4=True, Delta=.1)
H = mpctools.getCasadiFunc(meas, [Ncoef, Np], ["x", "p"], "H")

# Costo de etapa
def lfunc(w, v):
    return mpctools.mtimes(w.T, Q_inv, w) + mpctools.mtimes(v.T, R_inv, v)
l = mpctools.getCasadiFunc(lfunc, [Nw, Nv], ["w", "v"], "l")

# Costo de arrivo
def lxfunc(x, x0bar, Pinv):
    dx = x - x0bar
    if Ncoef == 1:
        return mpctools.mtimes(dx.T, dx) * Pinv
    else:
        return mpctools.mtimes(dx.T, Pinv, dx)
lx = mpctools.getCasadiFunc(lxfunc, [Ncoef, Ncoef, (Ncoef, Ncoef)], ["x", "x0bar", "Pinv"], "lx")

xhat = np.zeros((N, Ncoef))
yhat = np.zeros((N, Nv))
x0bar = np.zeros(Ncoef)
#x0bar = np.array([0,0,0,0,0]) # valor inicio al para el solver
guess = {}

#Definition of lower(lb) and upper(ub) bounds for coef estimations
#                Cd0, Cl_alpha, Cd2, Cn_p_alpha, Clp, Cm_alpha, Cm_p_alpha, Cm_q
x_lb = np.array([ 0,       0  ,   0, -np.inf, -np.inf,    -np.inf  , -np.inf, -np.inf])
x_ub = np.array([np.inf, np.inf, 10,     0,       0,         0 ,       np.inf,  0])
lb = {
    "x" : np.tile(x_lb, (Nt+1,1))
}
ub = {
    "x" : np.tile(x_ub, (Nt+1,1))
}

def ekf(f, h, x, u, w, y, p, P, Q, R, f_jacx=None, f_jacw=None, h_jacx=None):
    """
    EKF copiado del de mpctools, para adaptarlo a que H dependa de p también.

    Updates the prior distribution P^- using the Extended Kalman filter.

    f and h should be casadi functions. f must be discrete-time. P, Q, and R
    are the prior, state disturbance, and measurement noise covariances. Note
    that f must be f(x,u,w) and h must be h(x).

    If specified, f_jac and h_jac should be initialized jacobians. This saves
    some time if you're going to be calling this many times in a row, although
    it's really not noticable unless the models are very large. Note that they
    should return a single argument and can be created using
    mpctools.util.jacobianfunc.

    The value of x that should be fed is xhat(k | k-1), and the value of P
    should be P(k | k-1). xhat will be updated to xhat(k | k) and then advanced
    to xhat(k+1 | k), while P will be updated to P(k | k) and then advanced to
    P(k+1 | k). The return values are a list as follows

        [P(k+1 | k), xhat(k+1 | k), P(k | k), xhat(k | k)]

    Depending on your specific application, you will only be interested in
    some of these values.
    """

    # Check jacobians.
    if f_jacx is None:
        f_jacx = mpctools.util.jacobianfunc(f, 0)
    if f_jacw is None:
        f_jacw = mpctools.util.jacobianfunc(f, 2)
    if h_jacx is None:
        h_jacx = mpctools.util.jacobianfunc(h, 0)

    # Get linearization of measurement.
    C = np.array(h_jacx(x, p))
    yhat = np.array(h(x, p)).flatten()

    # Advance from x(k | k-1) to x(k | k).
    xhatm = x                                          # This is xhat(k | k-1)
    Pm = P                                             # This is P(k | k-1)
    L = linalg.solve(C.dot(Pm).dot(C.T) + R, C.dot(Pm)).T
    xhat = xhatm + L.dot(y - yhat)                     # This is xhat(k | k)
    P = (np.eye(Pm.shape[0]) - L.dot(C)).dot(Pm)       # This is P(k | k)

    # Now linearize the model at xhat.
    w = np.zeros(w.shape)
    A = np.array(f_jacx(xhat, u, w))
    G = np.array(f_jacw(xhat, u, w))

    # Advance.
    Pmp1 = A.dot(P).dot(A.T) + G.dot(Q).dot(G.T)       # This is P(k+1 | k)
    xhatmp1 = np.array(f(xhat, u, w)).flatten()     # This is xhat(k+1 | k)

    return [Pmp1, xhatmp1, P, xhat]

for k in range(N):
    N = {"x": Ncoef, "y": Ny, "p": Np, "u": Nu}
    N["t"] = min(k, Nt)
    tmin = max(0, k - Nt)
    tmax = k+1  # para que en los slice cuando ponga :tmax tome hasta k *inclusive*

    p_coefs = np.vstack((vt[tmin:tmax], alpha[tmin:tmax], beta[tmin:tmax], delta2[tmin:tmax], p[tmin:tmax], q[tmin:tmax],
                         r[tmin:tmax], rho[tmin:tmax])).T
    assert p_coefs.shape == (N["t"] + 1, Np)


    # Armo y llamo al solver. Si todavía no llené el horizonte, armo uno nuevo. Sino reúso el viejo., arma un problema de MHE de 1 a 10
    if k <= Nt:
        solver = mpctools.nmhe(f=F, h=H, u=np.zeros((tmax-tmin-1, Nu)), p=p_coefs,
                               y=F_body[tmin:tmax, :], l=l, N=N, lx=lx,
                               x0bar=x0bar, verbosity=0, guess=guess, lb=lb, ub=ub,
                               extrapar=dict(Pinv=linalg.inv(P)), inferargs=True)
    else:
        solver.par["Pinv"] = linalg.inv(P)
        solver.par["x0bar"] = x0bar
        #solver.saveguess()
        solver.par["y"] = list(F_body[tmin:tmax, :])
        solver.par["p"] = list(p_coefs)
    sol = mpctools.callSolver(solver)

    #print("x0bar",x0bar)
    #print("x0est",sol["x"][0])
    #input("Presione enter") # esto es para que pause y cont con (enter)

    if (k % 500) == 0:
        print(("%3d: %s" % (k, sol["status"])))
    if sol["status"] != "Solve_Succeeded":
        break

    xhat[k] = sol["x"][-1]  # xhat( t  | t )
    yhat[k] = np.squeeze(H(xhat[k], p_coefs[-1]))

    # Armo el guess.
    guess = {}
    for var in set(["x","w","v"]).intersection(sol.keys()):
        guess[var] = sol[var].copy()
    # Actualizo el guess
    if k + 1 > Nt:
        for key in guess.keys():
            guess[key] = guess[key][1:, ...]  # Tiro el guess más viejo

        # EKF para actualizar la matriz de covarianza P
        #[P, _, _, _] = ekf(F, H, x=sol["x"][0, :], u=np.zeros(Nu), w=sol["w"][0, ...], y=F_body[tmin], p=p_coefs[0], P=P, Q=Q, R=R)
        #print("P: ", P)
        x0bar = sol["x"][1]

    # Repito el último guess como guess del nuevo instante de tiempo
    for key in guess.keys():
        guess[key] = np.concatenate((guess[key], guess[key][-1:]))

#Save estimated coeficients
save_data(time, mach, xhat)

#Plot data vs mach
plot_data(Resul, data, mach, xhat)
#Plot data v time
plot_data_time(Resul, data, mach, xhat)
