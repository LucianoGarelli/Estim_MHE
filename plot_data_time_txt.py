# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as math

# Grafico para comparar la evolucion de los estados

#Read forces data
#resu1dir ='./Resultados/Nt_10_guess/'
resu2dir ='./Resultados/Nt_1_guess/'
resu3dir ='./Resultados/Nt_1/'
resu4dir ='./Resultados/Ref_coefs/'
resu5dir ='./Resultados/Nt_1_update_bounds/'
Resul = [resu2dir,resu3dir, resu5dir,resu4dir]
leg = ['Nt_1_g', 'Nt_1','Nt_1_u_b', 'Ref']

time = []
Cd0 = []
Cdd2 = []
CLa = []
Cn_p_alfa = []
Clp = []
Cm_alpha =[]
Cm_p_alpha = []
Cm_q = []

for k in range(np.size(Resul)):
    data = np.loadtxt(Resul[k]+'Coef_estim.txt', delimiter=',', skiprows=1)
    time.append(data[:,0])
    Cd0.append(data[:,2])
    Cdd2.append(data[:,4])
    CLa.append(data[:,3])
    Cn_p_alfa.append(data[:,5])
    Clp.append(data[:,6])
    Cm_alpha.append(data[:,7])
    Cm_p_alpha.append(data[:,8])
    Cm_q.append(data[:,9])

time = np.asarray(time).T
Cd0 = np.asarray(Cd0).T
Cdd2 = np.asarray(Cdd2).T
CLa = np.asarray(CLa).T
Cn_p_alfa = np.asarray(Cn_p_alfa).T
Clp = np.asarray(Clp).T
Cm_alpha = np.asarray(Cm_alpha).T
Cm_p_alpha = np.asarray(Cm_p_alpha).T
Cm_q = np.asarray(Cm_q).T
# Estimacion
# Grafico de coeficientes
fig_size = (12,4)
f, ax = plt.subplots(1,4, figsize=fig_size)
#Forces coefs
for k in range(np.size(Resul)):
    ax[0].plot(time[10:np.size(time),k], Cd0[10:np.size(time),k],label=leg[k])
    ax[1].plot(time[10:np.size(time),k], Cdd2[10:np.size(time),k])
    ax[2].plot(time[10:np.size(time),k], CLa[10:np.size(time),k])
    ax[3].plot(time[10:np.size(time),k], Cn_p_alfa[10:np.size(time),k])
ax[0].set_title('Cd0')
ax[1].set_title('Cdd2')
ax[2].set_title('CLa')
ax[3].set_title('Cn_p_alpha')
ax[0].legend()
#Moments coefs
f, ax = plt.subplots(1,4, figsize=fig_size)
for k in range(np.size(Resul)):
    ax[0].plot(time[10:np.size(time),k], Clp[10:np.size(time),k])
    ax[1].plot(time[10:np.size(time),k], Cm_alpha[10:np.size(time),k])
    ax[2].plot(time[10:np.size(time),k], Cm_p_alpha[10:np.size(time),k])
    ax[3].plot(time[10:np.size(time),k], Cm_q[10:np.size(time),k])
ax[0].set_title('Cl_p')
ax[1].set_title('Cm_alpha')
ax[2].set_title('Cm_p_alpha')
ax[3].set_title('Cm_q')

ax[0].legend()

plt.show()


