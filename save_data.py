# -*- coding: utf-8 -*-
import os
import numpy as np

def save_data(time, mach, x):

    MYDIR = ("Resultados")
    CHECK_FOLDER = os.path.isdir(MYDIR)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("Directorio Resultados creado: ", MYDIR)

    else:
        print(MYDIR, "Directorio Resultados existente.")

    #File to write force coeff
    ff = open("./Resultados/Coef_estim.txt", "w")  # xq le pasamos los las fuerzas de todo el CFD??
    ff.write(" # Time,      Mach,      Cd0,    Cl_alpha,    Cd2,  Cn_p_alpha,    Clp,   Cm_alpha,    Cm_p_alpha,    Cm_q \n")

    mach = mach.reshape(-1, 1)
    tt = time.reshape(-1, 1)
    coefs2= np.asarray([tt[:,0], mach[:,0]]).T
    coefs = np.concatenate((coefs2,x),axis=1)
    #np.column_stack(time, mach[:,0])
    #np.column_stack(time, x[:,0:8])
    np.savetxt(ff, coefs, delimiter=", ", fmt='%1.3e')
    ff.close()

    return