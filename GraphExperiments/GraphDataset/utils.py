import csv, os

def loadFromFile(filename,fileDelimiter=','):
    ##################
    # Load from file #
    ##################
    trans = []
    mx = my = mz = float("inf")
    nx = ny = nz = float("-inf")
    for line in csv.reader(file(os.path.expanduser(filename)),delimiter=fileDelimiter):
        t = [float(l) for l in line[:-1]] + [line[-1]]
        t, r, imgName = t[:3],t[3:-1], t[-1]

        alpha, gama, beta  = r
        x, z, y = t
        trans.append([x, y, z, alpha, beta, gama, imgName])

        mx = min(x,mx)
        nx = max(x,nx)
        my = min(y,my)
        ny = max(y,ny)
        mz = min(z,mz)
        nz = max(z,nz)

    ra = max(nx-mx, ny-my)
    mx , nx = mx + (nx-mx)/2. - ra/2. - 0.1*ra, mx + (nx-mx)/2. + ra/2. + 0.1*ra
    my , ny = my + (ny-my)/2. - ra/2. - 0.1*ra,my + (ny-my)/2. + ra/2.  + 0.1*ra
    return trans, (mx , nx), (my , ny), (mz, nz)
