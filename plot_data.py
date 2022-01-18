# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as math

# Grafico la evolucion de los estados

def plot_data(Resul, data, mach, x):

    #Read forces data
    data_ref = np.loadtxt(Resul[0]+'Force_coef_proc.txt', delimiter=',', skiprows=1)
    mach_ref = data_ref[:,1]
    Cd0_ref = data_ref[:,5]
    Cdd2_ref = data_ref[:,6]
    CLa_ref = data_ref[:,7]
    Cn_p_alfa = data_ref[:,8]

    fig_size = (15,5)
    leg = ['Estim', 'Used', 'Alpha', 'Beta', 'Tot']
    time = data[:,0]
    alpha = data[:, 1]
    beta = data[:, 2]
    delta2 = ((np.sin(beta))**2 + (np.cos(beta))**2*(np.sin(alpha))**2)

    # Estimacion
    # Grafico de coeficientes
    f, ax = plt.subplots(1,6, figsize=fig_size)
    #for k in range(3):
    ax[0].plot(mach, x[:,0],label=leg[0])
    ax[0].plot(mach_ref, Cd0_ref,label=leg[1])
    ax[1].plot(mach, x[:,2])
    ax[1].plot(mach_ref, Cdd2_ref)
    ax[2].plot(time, alpha * 180 / np.pi, label=leg[2])
    ax[2].plot(time, beta * 180 / np.pi, label=leg[3])
    ax[2].plot(time, np.sqrt(delta2) * 180 / np.pi, label=leg[4])
    ax[0].legend()
    ax[2].legend()

    ax[3].plot(mach, x[:,1])
    ax[3].plot(mach_ref, CLa_ref)

    ax[4].plot(mach, x[:,0] + x[:,2]*delta2)
    ax[4].plot(mach_ref, Cd0_ref + Cdd2_ref*delta2)

    ax[5].plot(mach, x[:,3])
    ax[5].plot(mach_ref, Cn_p_alfa)

    ax[0].set_title('Cd_0 vs Mach')
    ax[1].set_title('Cd2 vs Mach')
    ax[2].set_title('Ang[deg] vs Mach')
    ax[3].set_title('Cd vs Mach')
    ax[4].set_title('Cl_alfa vs Mach')
    ax[5].set_title('Cn_p_alfa vs Mach')

    #Read moments data
    data_ref = np.loadtxt(Resul[0]+'Moment_coef_proc.txt', delimiter=',', skiprows=1)
    mach_ref = data_ref[:,1]
    Clp_ref = data_ref[:,4]
    Cm_alfa_ref = data_ref[:,5]
    Cm_p_alfa_ref = data_ref[:,6]
    Cm_q_ref = data_ref[:,7]
    Cn_beta_ref = data_ref[:,8]
    Cn_r_ref = data_ref[:,9]

    fig_size = (12,4)
    time = data[:,0]
    alpha = data[:, 1]
    beta = data[:, 2]

    # Estimacion
    # Grafico de coeficientes
    f, ax = plt.subplots(1,4, figsize=fig_size)
    #for k in range(3):
    ax[0].plot(mach, x[:,4])
    ax[0].plot(mach_ref, Clp_ref)
    ax[1].plot(mach, x[:,5])
    ax[1].plot(mach_ref, Cm_alfa_ref)
    ax[2].plot(mach, x[:,6])
    ax[2].plot(mach_ref, Cm_p_alfa_ref)

    ax[3].plot(mach, x[:,7])
    ax[3].plot(mach_ref, Cm_q_ref)

    #ax[4].plot(mach, x[:,3])
    #ax[4].plot(mach_ref, Cn_beta_ref)

    ax[0].set_title('Clp vs Mach')
    ax[1].set_title('Cm_alfa_ref vs Mach')
    ax[2].set_title('Cm_p_alfa_ref vs Mach')
    ax[3].set_title('Cm_q_ref vs Mach')
    #ax[4].set_title('Cn_beta_ref vs Mach')


    plt.show(block=False)
    return