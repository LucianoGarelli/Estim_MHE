# -*- coding: utf-8 -*-
"""

@author: lgenzelis, ntrivisonno
"""

import numpy as np
import matplotlib.pyplot as plt
import mpctools
import casadi
from scipy import linalg

diam = 0.04
rho = 1.225
S = np.pi * (0.5 * diam) ** 2

CASO = 0

# carga dato una sola corrida
#
#data = np.loadtxt('Resu_RBD/' + ['Forces_proc_C01a0_25.txt'][CASO], delimiter=',', skiprows=1)
data = np.loadtxt('Resu_RBD/' + 'Forces_proc_C_B03.txt', delimiter=',', skiprows=1)
#
# carga datos, solo drag
#data = np.loadtxt('Resu_RBD/' + ['Forces_proc_B03.txt', 'Forces_proc_B04.txt', 'Forces_proc_B06.txt'][CASO], delimiter=',', skiprows=1) #mail_nicolas/ hace alusion a la carpeta | skiprows=1 saltea la primer fila xq es el encabezado
#
# carga datos, drag + lift
#data = np.loadtxt('Resu_RBD/' + ['Forces_proc_C_C01.txt', 'Forces_proc_C_C02.txt', 'Forces_proc_C_C03.txt'][CASO], delimiter=',', skiprows=1) #mail_nicolas/ hace alusion a la carpeta | skiprows=1 saltea la primer fila xq es el encabezado
N = data.shape[0]

print(data.shape)

# Encabezado del txt:
# Time, alpha, beta, delta2, V_inf (= V_t), u(v_body_X), v(v_body_Y), w(v_body_Z), p, q, r, gx, gy, gz, FX_body, FY_body, FZ_body

alpha = data[:, 1]
beta = data[:, 2]
delta2 = data[:, 3]  # alpha2
vt = data[:, 4]
u = data[:, 5]  # vel_body_X
v = data[:, 6]  # vel_body_Y
w = data[:, 7]  # vel_body_Z
p = data[:, 8]
q = data[:, 9]
r = data[:, 10]
grav = data[:, 11:14]  # gx, gy, gz
F_body = data[:, 13:16]  # FX_body, FY_body, FZ_body

# F_body = F_body - grav  # FIXME

Ncoef = 3 # cant de coef a estimar
Ny = 3
Nw = Ncoef
Nv = Ny # cant ruido medicion, simil cant mediciones
Np = 4 # cant de parametros al solver
Nt = 10  # horizonte
Nu = 0

Q = np.diag([100.] * Ncoef)  # matrix de covarianza de ruido de proceso
R = np.diag([.1, .1, .1])     # matrix de covarianza de ruido de  medición
P = np.diag([10.] * Ncoef)    # matrix de covarianza de estimación inicial

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
    p[0]: vt
    p[1]: alpha
    p[2]: beta
    p[3]: delta2
    y[0]: Fx
    y[1]: Fy
    y[2]: Fz
    '''
    qdy = 0.5 * rho * p[0] ** 2
    y = casadi.SX.zeros(Ny)
    y[0] = -qdy * S * np.cos(p[1]) * np.cos(p[2]) * (x[0]+x[2]*p[3]) + qdy * S * x[1] * (1-(np.cos(p[1])**2 * np.cos(p[2])**2))
    y[1] = -qdy * S * np.sin(p[2]) * (x[0]+x[2]*p[3]) - qdy * S * x[1] * np.cos(p[1]) * np.cos(p[2]) * np.sin(p[2])
    y[2] = -qdy * S * np.sin(p[1]) * np.cos(p[2]) * (x[0]+x[2]*p[3]) + qdy * S * x[1] * np.cos(p[1]) * np.sin(p[1]) * np.cos(p[2])**2
    assert Ny == 3
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
#x0bar = np.array([200,10]) # valor inicio al para el solver
guess = {}


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

    p_coefs = np.vstack((vt[tmin:tmax], alpha[tmin:tmax], beta[tmin:tmax], delta2[tmin:tmax])).T
    assert p_coefs.shape == (N["t"] + 1, Np)


    # Armo y llamo al solver. Si todavía no llené el horizonte, armo uno nuevo. Sino reúso el viejo., arma un problema de MHE de 1 a 10
    if k <= Nt:
        solver = mpctools.nmhe(f=F, h=H, u=np.zeros((tmax-tmin-1, Nu)), p=p_coefs,
                               y=F_body[tmin:tmax, :], l=l, N=N, lx=lx,
                               x0bar=x0bar, verbosity=0, guess=guess,
                               extrapar=dict(Pinv=linalg.inv(P)), inferargs=True)
    else:
        solver.par["Pinv"] = linalg.inv(P)
        solver.par["x0bar"] = x0bar
        
        solver.par["y"] = list(F_body[tmin:tmax, :])
        solver.par["p"] = list(p_coefs)
    sol = mpctools.callSolver(solver)

    print("x0bar",x0bar)
    print("x0est",sol["x"][0])
    #input("Presione enter") # esto es para que pause y cont con (enter)


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
        [P, _, _, _] = ekf(F, H, x=sol["x"][0, :], u=np.zeros(Nu), w=sol["w"][0, ...], y=F_body[tmin], p=p_coefs[0], P=P, Q=Q, R=R)
        print("P: ", P)
        x0bar = sol["x"][1]

    # Repito el último guess como guess del nuevo instante de tiempo
    for key in guess.keys():
        guess[key] = np.concatenate((guess[key], guess[key][-1:]))

Cd0_estim = xhat[:, 0]
Cl_estim = xhat[:, 1]
Cd2_estim = xhat[:, 2]

Cd_estim = Cd0_estim + Cd2_estim * delta2

#
# carga datos 1 sola corrida
#coefs_reales = np.loadtxt('Resu_RBD/' + ['Force_coef_proc_C01a0_25.txt'][CASO], delimiter=',', skiprows=1)

# Encabezado del txt:
# Time,   Mach,     alfa,     beta,     delta2,     Cd,     CL_alfa,     Cn_p_alfa,     Cn_q_alfa

#t = coefs_reales[:, 0]
#mach = coefs_reales[:, 1]
#Cd_real = coefs_reales[:, 5]
#Cl_real = coefs_reales[:, 6]


#
# Cd
#

# estimacion
# Grafico el coeficiente
f, ax = plt.subplots(3)
#ax[0].plot(mach, Cd_real,'o', label='Cd Real')
ax[0].plot(mach, Cd_estim, label='Cd Estimado')
ax[0].legend()
ax[0].set_xlim([min(mach), max(mach)])
ax[0].set_title('Cd vs Mach')

# me fijo tamanos
print("%------------------------------%")
print("size(t): ", np.size(t))
print("size(mach): ", np.size(mach))
print("%------------------------------%")

# queriamos hacer una funcion para el secondary_xaxis
def cdt(x):
    return 

#ax[1].plot(t, Cd_real,'o', label='Cd Real')
ax[1].plot(t, Cd_estim, label='Cd Estimado')
ax[1].legend()
ax[1].set_xlim([0, max(t)])
ax[1].set_title('Cd vs tiempo')
#secax = ax[1].secondary_xaxis('top', mach)
#secax.set_xlabel('Mach')


ax[2].plot(mach, Cd_estim, label='Cd Estim')
ax[2].plot(mach, Cd0_estim, label='Cd0')
ax[2].plot(mach, Cd2_estim*delta2, label='Cdd2*delta2')
ax[2].legend()
ax[2].set_xlim([min(mach), max(mach)])
ax[2].set_title('Cd vs Mach')

plt.tight_layout()
plt.savefig('Figures/CD - ' + ['Caso 8', 'Caso 10', 'Caso 11'][CASO] + '.png', bbox_inches='tight')

## Grafico errores
f, ax = plt.subplots(2)
ax[0].plot(mach, Cd_real - Cd_estim)
ax[0].set_xlim([min(mach), max(mach)])
ax[0].set_title('Error en Cd vs Mach')

ax[1].plot(t, Cd_real - Cd_estim)
ax[1].set_xlim([0, max(t)])
ax[1].set_title('Error en Cd vs tiempo')

plt.tight_layout()
plt.savefig('Figures/CD_error - ' + ['Caso 8', 'Caso 10', 'Caso 11'][CASO] + '.png', bbox_inches='tight')
'''
#
# Cl
#
# Grafico el coeficiente
f, ax = plt.subplots(2)
ax[0].plot(mach, Cl_real, label='Cl Real')
ax[0].plot(mach, Cl_estim, label='Cl Estimado')
ax[0].legend()
ax[0].set_xlim([min(mach), max(mach)])
ax[0].set_title('Cl vs Mach')

ax[1].plot(t, Cl_real, label='Cl Real')
ax[1].plot(t, Cl_estim, label='Cl Estimado')
ax[1].legend()
ax[1].set_xlim([0, max(t)])
ax[1].set_title('Cl vs tiempo')

plt.tight_layout()
plt.savefig('Figures/Cl - ' + ['Caso 8', 'Caso 10', 'Caso 11'][CASO] + '.png', bbox_inches='tight')

## Grafico errores 
f, ax = plt.subplots(2)
ax[0].plot(mach, Cl_real - Cl_estim)
ax[0].set_xlim([min(mach), max(mach)])
ax[0].set_title('Error en Cl vs Mach')

ax[1].plot(t, Cl_real - Cl_estim)
ax[1].set_xlim([0, max(t)])
ax[1].set_title('Error en Cl vs tiempo')

plt.tight_layout()
plt.savefig('Figures/Cl_error - ' + ['Caso 8', 'Caso 10', 'Caso 11'][CASO] + '.png', bbox_inches='tight')

#plt.show()
'''

plt.show()
