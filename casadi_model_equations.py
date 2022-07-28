import casadi as cs


def meas(_x, _p, _S, _diam):

    qdy = 0.5 * _p[7] * _p[0] ** 2
    # y = cs.SX.zeros(Ny)

    ca = cs.cos(_p[1])
    sa = cs.sin(_p[1])
    cb = cs.cos(_p[2])
    sb = cs.sin(_p[2])
    Cd = _x[0] + _x[2] * _p[3]
    CL_alfa = _x[1]
    Cn_p_alfa = _x[3]
    Clp = _x[4]
    Cm_alfa = _x[5]
    Cm_p_alfa = _x[6]
    Cm_q = _x[7]
    #Forces
    # y[0] = qdy*S*(-Cd*ca*cb + CL_alfa*(sb**2 + sa**2 * cb**2))
    # y[1] = qdy*S*(-Cd * sb - CL_alfa * (ca*sb*cb) - Cn_p_alfa * _p[4] * diam * (sa * cb) / _p[0])
    # y[2] = qdy*S*(-Cd * sa * cb - CL_alfa * (sa*ca*cb**2) + Cn_p_alfa * _p[4] * diam * sb / _p[0])
    # #Moments
    # y[3] = qdy * S * diam * (_p[4] * diam / _p[0]) * Clp
    # y[4] = qdy*S*diam*(Cm_alfa * (sa * cb) - (_p[4] * diam / _p[0]) * (-Cm_p_alfa) * sb + (diam / _p[0]) * Cm_q * _p[5])
    # y[5] = qdy*S*diam*(-Cm_alfa * sb - (_p[4] * diam / _p[0]) * (-Cm_p_alfa) * (sa * cb) + (diam / _p[0]) * Cm_q * _p[6])
    # assert Ny == 6
    # return [y[0], y[1], y[2], y[3], y[4], y[5]]
    # print(qdy * _S * (-Cd * ca * cb + CL_alfa * (sb ** 2 + sa ** 2 * cb ** 2)))
    return [qdy * _S * (-Cd * ca * cb + CL_alfa * (sb ** 2 + sa ** 2 * cb ** 2)),
            qdy * _S * (-Cd * sb - CL_alfa * (ca * sb * cb) - Cn_p_alfa * _p[4] * _diam * (sa * cb) / _p[0]),
            qdy * _S * (-Cd * sa * cb - CL_alfa * (sa * ca * cb ** 2) + Cn_p_alfa * _p[4] * _diam * sb / _p[0]),
            qdy * _S * _diam * (_p[4] * _diam / _p[0]) * Clp,
            qdy * _S * _diam * (Cm_alfa * (sa * cb) - (_p[4] * _diam / _p[0]) * (-Cm_p_alfa) * sb + (_diam / _p[0]) * Cm_q * _p[5]),
            qdy * _S * _diam * (-Cm_alfa * sb - (_p[4] * _diam / _p[0]) * (-Cm_p_alfa) * (sa * cb) + (_diam / _p[0]) * Cm_q * _p[6])]