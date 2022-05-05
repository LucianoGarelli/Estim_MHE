# -*- coding: utf-8 -*-
"""

@author: lgenzelis, ntrivisonno, gsanchez, lgarelli
"""
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import mpctools as mpc
import h5py

from parameters import parameters
from fluid_prop import fluid_prop
from plot_data import plot_data
from plot_data_time import plot_data_time
from save_data import  save_data
from plot_data_noise import plot_data_noise

m, diam, xcg, ycg, zcg, Ixx, Iyy, Izz, steps, dt = parameters('./Data/data_F01.dat')

S = np.pi * (0.5 * diam) ** 2
g = 9.81  # aceleración de la gravedad

# Path to data
Resul = ['/home/ntrivi/Documents/Tesis/RBD/Resu_ref/Wernert_AIAA2010_7460/Caso_F01/'] 
#Resul = ['/home/ntrivi/Documents/Tesis/RBD/Resu_ref/Wernert_AIAA2010_7460/Caso_F01_unificated/']

#Read forces-moments data
data = np.loadtxt(Resul[0]+'Forces_proc.txt', delimiter=',', skiprows=1)
datam = np.loadtxt(Resul[0]+'Moments_proc.txt', delimiter=',', skiprows=1)
Nsim = data.shape[0]
xned = []
h5f = h5py.File(Resul[0]+'Data.hdf5','r')
# Read data from hdf5
#xned.append(h5f['/Inertial_coord'][:,:])
xned.append(h5f['/Inertial_coord'][1:Nsim+1,:])
h5f.close()

# Propiedades fluido vs altura
#rho, mu, c = fluid_prop(xned[0][1:N+1,2], 0)
#rho, mu, c = fluid_prop(xned[0][1:N+1,2], 0)
rho, mu, c = fluid_prop(xned[0][:,2], 0)

print(data.shape)

# Encabezado del txt:
# Time, alpha, beta, V_inf (= V_t), u(v_body_X), v(v_body_Y), w(v_body_Z), p, q, r, gx, gy, gz, FX, FY, FZ

time = data[:,0]
alpha = data[:, 1]
beta = data[:, 2]
delta2 = ((np.sin(beta))**2 + (np.cos(beta))**2*(np.sin(alpha))**2) # alpha2
vt = data[:,3]
mach = vt/c
u = data[:, 4]  # vel_body_X
v = data[:, 5]  # vel_body_Y
w = data[:, 6]  # vel_body_Z
p = data[:, 7]
q = data[:, 8]
r = data[:, 9]
grav = data[:, 10:13]  # gx, gy, gz
F_body = data[:, 13:16]  # FX, FY, FZ
F_body = np.column_stack([F_body,datam[:, 6:9]]) #Add moments to F_body

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

Nx = 8 # Number of coef wanted to estimate
Nu = 0
Nw = Nx
Nz = 2
Np = 8  # Number de parameters given to solver
Ny = 6
Nv = Ny # Number of mesuarements noise, equal number of mesuarements
Delta = 0.01
Nc = 3  # Number of collocation points
Nt = 20 # Horizon windows

# Explicit z function
def zfunc(x, p):
    return [x[0] + x[2] * p[3], 0.5 * p[7] * p[0] ** 2]

# ODE for VPO with and without z
def odefunc(x, u, z, p, w):
    return 0. + w

# Algebraic function for implicit solution of z
def gfunc(x,z,p):
    return z - zfunc(x, p)

# Measurement function
def hmeas(x, z, p):
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
    qdy = z[1]  # 0.5 * p[7] * p[0] ** 2
    y = cs.SX.zeros(Ny)

    ca = np.cos(p[1])
    sa = np.sin(p[1])
    cb = np.cos(p[2])
    sb = np.sin(p[2])
    Cd = z[0]  # x[0] + x[2] * p[3]
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

# Get casadi functions for ODE and DAE
# fode = mpc.getCasadiFunc(odefunc,[Nx,Nu],["x","u"],funcname="fode")
fdae = mpc.getCasadiFunc(odefunc, [Nx, Nu, Nz, Np, Nw],["x", "u", "z", "p", "w"], funcname="fdae")
gdae = mpc.getCasadiFunc(gfunc, [Nx, Nz, Np],["x", "z", "p"], funcname="gdae")
hmeas = mpc.getCasadiFunc(hmeas, [Nx, Nz, Np], ["x", "z", "p"], "hmeas")


Q = np.eye(Nx)
R = np.eye(Ny)
Pinv = np.eye(Nx)

# Define stage cost
def stagecost(w, v):
    return mpc.mtimes(w.T, Q, w) + mpc.mtimes(v.T, R, v)
l = mpc.getCasadiFunc(stagecost, [Nw, Nv], ["w", "v"], "l")

# Define arrival cost
def arrivalcost(x, x0bar, Pinv):
    dx = x - x0bar
    return mpc.mtimes(dx.T, Pinv, dx)
lx = mpc.getCasadiFunc(arrivalcost, [Nx, Nx, (Nx, Nx)], ["x", "x0bar", "Pinv"], "lx")

# Size of variables
N = {"x":Nx, "z":Nz, "p":Np, 'c':Nc, "t":Nt, "u": Nu, "y": Ny}

# Initial state
# x0 = np.array([0,0,0,0,0,0,0,0])
x0bar = np.array([ 2.96736843e-01,  2.74668167e+00,  3.45, -7.66722171e-01, -1.04984336e-02, -3.16870955e+00, -4.80740759e-01, -14])
# Bounds on optimization variables
x_lb = np.array([ 0,       0  ,   0, -np.inf, -np.inf,    -np.inf  , -np.inf, -np.inf])
x_ub = np.array([np.inf, np.inf, 10,     0,       0,         0 ,       np.inf,  0])
z_lb = np.array([0, -np.inf])
z_ub = np.array([np.inf, np.inf])
w_lb = np.zeros((Nw,))
w_ub = np.zeros((Nw,))
lb = {
    "x" : np.tile(x_lb, (Nt+1,1)),
    "z" : np.tile(z_lb, (Nt+1,1)),
    "w" : np.tile(w_lb, (Nt,1))
}
ub = {
    "x" : np.tile(x_ub, (Nt+1,1)),
    "z" : np.tile(z_ub, (Nt+1,1)),
    "w" : np.tile(w_ub, (Nt,1))
}

# ODE arguements

xhat = np.zeros((Nsim, Nx))
yhat = np.zeros((Nsim, Ny))
zhat = np.zeros((Nsim, Nz))

# solver = mpc.nmhe(**nmheargs)
# sol_dae = mpc.callSolver(solver)
# xhat[tmin:tmax] = sol_dae['x']

for k in range(Nsim):
    # N = {"x": Nx, "y": Ny, "p": Np, "u": Nu}
    N["t"] = min(k, Nt)
    tmin = max(0, k - Nt)
    tmax = k+1  # para que en los slice cuando ponga :tmax tome hasta k *inclusive*

    p_coefs = np.vstack((vt[tmin:tmax], alpha[tmin:tmax], beta[tmin:tmax], delta2[tmin:tmax], p[tmin:tmax], q[tmin:tmax],
                         r[tmin:tmax], rho[tmin:tmax])).T
    assert p_coefs.shape == (N["t"] + 1, Np)


    # Armo y llamo al solver. Si todavía no llené el horizonte, armo uno nuevo. Sino reúso el viejo, arma un problema de MHE de 1 a 10
    if k <= Nt:
        nmheargs = {
            "f": fdae,
            "g": gdae,
            "h": hmeas,
            "u": np.zeros((tmax - tmin - 1, Nu)),
            "p": p_coefs,
            "y": F_body[tmin:tmax, :],
            "l": l,
            "lx": lx,
            "N": N,
            "x0bar": x0bar,
            "lb": lb,
            "ub": ub,
            "Delta": Delta,
            "extrapar": dict(Pinv=Pinv),
            "inferargs": True,
            "verbosity": 0
        }

        # nmheargs["u"] = np.zeros((tmax - tmin - 1, Nu))
        # nmheargs["p"] = p_coefs
        # nmheargs["y"] = F_body[tmin:tmax, :]
        # solver = mpc.nmhe(f=F, h=H, u=np.zeros((tmax-tmin-1, Nu)), p=p_coefs,
        #                        y=F_body[tmin:tmax, :], l=l, N=N, lx=lx,
        #                        x0bar=x0bar, verbosity=0, guess=guess, lb=lb, ub=ub,
        #                        extrapar=dict(Pinv=linalg.inv(P)), inferargs=True)
        solver = mpc.nmhe(**nmheargs)
    else:
        solver.par["Pinv"] = Pinv
        solver.par["x0bar"] = x0bar
        #solver.saveguess()
        solver.par["y"] = list(F_body[tmin:tmax, :])
        solver.par["p"] = list(p_coefs)
    sol = mpc.callSolver(solver)

    #print("x0bar",x0bar)
    #print("x0est",sol["x"][0])
    #input("Presione enter") # esto es para que pause y cont con (enter)

    # print(("%3d: %s" % (k, sol["status"])))
    if (k % 500) == 0:
        print(("%3d: %s" % (k, sol["status"])))
    if sol["status"] != "Solve_Succeeded":
        break

    xhat[k] = sol["x"][-1]  # xhat( t  | t )
    zhat[k] = sol["z"][-1]
    yhat[k] = np.squeeze(hmeas(xhat[k], zhat[k], p_coefs[-1]))

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

# Solve and plot for initial horizon
# sol_ode = mpc.callSolver(solvers["ode"])
# sol_dae = mpc.callSolver(solvers["dae"])

x_ = cs.SX.sym('X', Nx)  # Differential states
u_ = cs.SX.sym('U', Nu)  # Fictitious control
w_ = cs.SX.sym('W', Nw)  # Differential states
z_ = cs.SX.sym('Z', Nz)  # Algebraic states
p_ = cs.SX.sym('P', Np)  # System parameters
y_ = cs.SX.sym('Y', Ny)  # Measurements
print("fdae(x,z,p,w): ", fdae(x_,u_,z_,p_,w_))
print("gdae(x,z,p): ", gdae(x_,z_,p_))

#Save estimated coeficients
save_data(time, mach, alpha, xhat)

#Plot data vs mach
plot_data(Resul, data, mach, xhat)
#Plot data v time
plot_data_time(Resul, data, mach, xhat)
