# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as math

# Grafico la evolucion de los estados

def plot_data_time(Resul, data, mach, x):

    #Read forces data
    data_ref = np.loadtxt(Resul[0]+'Force_coef_proc.txt', delimiter=',', skiprows=1)
    time_ref = data_ref[:,0]
    alpha_ref = data_ref[:, 1]
    beta_ref = data_ref[:, 2]
    Cd0_ref = data_ref[:,5]
    Cdd2_ref = data_ref[:,6]
    CLa_ref = data_ref[:,7]
    Cn_p_alfa = data_ref[:,8]
    delta2_ref = ((np.sin(beta_ref))**2 + (np.cos(beta_ref))**2*(np.sin(alpha_ref))**2)

    fig_size = (15,5)
    leg = ['Estim', 'Used','Alpha','Beta','Tot']
    time = data[:,0]
    alpha = data[:, 1]
    beta = data[:, 2]
    delta2 = ((np.sin(beta))**2 + (np.cos(beta))**2*(np.sin(alpha))**2)

    # Estimacion
    # Grafico de coeficientes
    f, ax = plt.subplots(1,6, figsize=fig_size)
    #for k in range(3):
    ax[0].plot(time, x[:,0],label=leg[0])
    ax[0].plot(time_ref, Cd0_ref,label=leg[1])
    ax[1].plot(time, x[:,2])
    ax[1].plot(time_ref, Cdd2_ref)
    ax[2].plot(time, alpha*180/np.pi,label=leg[2])
    ax[2].plot(time, beta*180/np.pi,label=leg[3])
    ax[2].plot(time, np.sqrt(delta2)*180/np.pi,label=leg[4])
    ax[0].legend()
    ax[2].legend()

    ax[4].plot(time, x[:,1])
    ax[4].plot(time_ref, CLa_ref)

    ax[3].plot(time, x[:,0] + x[:,2]*delta2)
    ax[3].plot(time_ref, Cd0_ref + Cdd2_ref*delta2_ref)

    ax[5].plot(time, x[:,3])
    ax[5].plot(time_ref, Cn_p_alfa)

    ax[0].set_title('Cd_0 vs time')
    ax[1].set_title('Cd2 vs time')
    ax[2].set_title('Ang[deg] vs time')
    ax[3].set_title('Cd vs time')
    ax[4].set_title('Cl_alfa vs time')
    ax[5].set_title('Cn_p_alfa vs time')

    #Read moments data
    data_ref = np.loadtxt(Resul[0]+'Moment_coef_proc.txt', delimiter=',', skiprows=1)
    time_ref = data_ref[:,0]
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
    ax[0].plot(time, x[:,4],label=leg[0])
    ax[0].plot(time_ref, Clp_ref,label=leg[1])
    ax[0].legend()
    ax[1].plot(time, x[:,5])
    ax[1].plot(time_ref, Cm_alfa_ref)
    ax[2].plot(time, x[:,6])
    ax[2].plot(time_ref, Cm_p_alfa_ref)

    ax[3].plot(time, x[:,7])
    ax[3].plot(time_ref, Cm_q_ref)

    #ax[4].plot(time, x[:,8])
    #ax[4].plot(time_ref, Cn_beta_ref)

    ax[0].set_title('Clp vs time')
    ax[1].set_title('Cm_alfa_ref vs time')
    ax[2].set_title('Cm_p_alfa_ref vs time')
    ax[3].set_title('Cm_q_ref vs time')
    #ax[4].set_title('Cn_beta_ref vs time')


    plt.show()
    return