# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# Grafico la evolucion de los estados

def plot_data_noise(time, F_body, noise):

    fig_size = (15,5)
    F_body_n = F_body + noise
    # Estimacion
    # Grafico de coeficientes
    f, ax = plt.subplots(1,3, figsize=fig_size)
    for k in range(3):
        ax[k].plot(time, F_body[:,k])
        ax[k].plot(time, F_body_n[:, k])

    ax[0].set_title('Fx vs Time')
    ax[1].set_title('Fy vs Time')
    ax[2].set_title('Fz vs Time')

    f, ax = plt.subplots(1, 3, figsize=fig_size)
    for k in range(3):
        ax[k].plot(time, F_body[:, k+3])
        ax[k].plot(time, F_body_n[:, k+3])

    ax[0].set_title('Mx vs Time')
    ax[1].set_title('My vs Time')
    ax[2].set_title('Mz vs Time')

    plt.show(block=False)
    return