def parameters(file_in):
    f = open(file_in, 'r')
    # Lectura comentario
    line = f.readline()
    line = f.readline()
    frags = line.split()
    masa = float(frags[1])

    line = f.readline()
    frags = line.split()
    diam = float(frags[1])

    line = f.readline()
    frags = line.split()
    xcg = float(frags[1])
    ycg = float(frags[3])
    zcg = float(frags[5])

    line = f.readline()
    frags = line.split()
    Ixx = float(frags[1])
    Iyy = float(frags[3])
    Izz = float(frags[5])

    line = f.readline()
    frags = line.split()
    steps = int(frags[1])

    line = f.readline()
    frags = line.split()
    dt = float(frags[1])

    return [masa, diam, xcg, ycg, zcg, Ixx, Iyy, Izz, steps, dt]
