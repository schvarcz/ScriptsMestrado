    # -*- coding: utf8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm, imread
from matplotlib.gridspec import GridSpec
import numpy as np
import csv, os
from numpy import *
from math import atan2,pi


def matrixRot(alpha,beta,gama):
    return matrix( 
    [ [+cos(beta)*cos(gama), -cos(beta)*sin(gama), +sin(beta)],
    [+sin(alpha)*sin(beta)*cos(gama)+cos(alpha)*sin(gama),-sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama),-sin(alpha)*cos(beta)],
    [-cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama), +cos(alpha)*sin(beta)*sin(gama)+sin(alpha)*cos(gama), +cos(alpha)*cos(beta)]]) 



paths = [ 
#            "2010_03_09_drive_0019",
#            "drone/20140318_132620",
#            "drone/20140318_132620_gray",
#            "drone/20140318_133931",
#            "drone/20140318_133931_gray",
#            "drone/20140327_135316_gray",
#            "drone/20140328_102444_gray",
#            "nao/nao2",
#            "nao/nao2_gray",
#            "nao/nao2_rect",
            "nao/nao2_rect_escolha",
#            "nao/naooo_2014-03-10-17-48-35",
#            "nao/naooo_2014-03-10-17-48-35_gray"
            ]

steps = [1]#,2,3,5,10,15,20,25]

for path in paths:
    pathFull = os.path.expanduser("~/Dissertacao/datasets/"+path)
    for step in steps:
        transLibViso = [[],[],[]]
        #oldR = asarray([.1,0.,0.]) #rotação da camera em relação ao gps
        oldR = asarray([0.,0.,0.]) #rotação da camera em relação ao gps
        mx = my = float("inf")
        nx = ny = float("-inf")
        angles = [0.]
        for line in csv.reader(file(pathFull+"/posicoes_{}.csv".format(step)),delimiter=','):
            t = [float(l) for l in line]

            alpha, beta, gama = oldR
            t = matrixRot(alpha,beta,gama).I.dot(t).A1

            x, y, z = list(t)
            angles.append(atan2(z,x))
            
            transLibViso[0].append(x)
            transLibViso[1].append(y)
            transLibViso[2].append(z)
            mx = min(x,mx)
            nx = max(x,nx)
            my = min(z,my)
            ny = max(z,ny)

        ra = max(nx-mx, ny-my)
        mx , nx = mx + (nx-mx)/2. - ra/2. - 0.1*ra, mx + (nx-mx)/2. + ra/2. + 0.1*ra
        my , ny = my + (ny-my)/2. - ra/2. - 0.1*ra,my + (ny-my)/2. + ra/2.  + 0.1*ra

        i = 1
        print path, step
        i += step

        f = plt.figure()
        gs = GridSpec(3,2)
        #ax1 = plt.subplot(gs[0,1:])
        ax2 = plt.subplot(gs[:-1,1])
        ax3 = plt.subplot(gs[0:,0])
        ax4 = plt.subplot(gs[-1,1])
        #f, ((ax1,ax2),(ax3,ax4)) = subplots(2,2)
        img = imread(os.path.expanduser("~/Dissertacao/OdometriaVisual/features/{0}/step_{1}/I1_{2:06d}.png".format(path,step,10)))
        ax2.imshow(img)

        ax3.clear()
        ax3.plot(transLibViso[0],transLibViso[2],"b")
        ax3.set_xlabel("X Axis (m)")
        ax3.set_ylabel("Y Axis (m)")
        ax3.set_xlim(mx,nx)
        ax3.set_ylim(my,ny)
        
        ax4.plot(angles)
        plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        plt.show()


