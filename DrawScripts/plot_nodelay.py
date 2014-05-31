# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm, imread
from matplotlib.gridspec import GridSpec
import numpy as np
import csv, os
from numpy import *


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
            "drone/20140328_102444_gray",
#            "nao/nao2",
#            "nao/nao2_gray",
#            "nao/naooo_2014-03-10-17-48-35",
#            "nao/naooo_2014-03-10-17-48-35_gray"
            ]

steps = [1]#,2,3,5,10,15,20,25]

for path in paths:
    for step in steps:
        transLibViso = [[],[],[]]
        #oldR = asarray([.1,0.,0.]) #rotação da camera em relação ao gps
        oldR = asarray([0.1,0.,0.5]) #rotação da camera em relação ao gps
        mx = my = float("inf")
        nx = ny = float("-inf")
        for line in csv.reader(file(path+"/posicoes_{}.csv".format(step)),delimiter=','):
            t = [float(l) for l in line]

            alpha, beta, gama = oldR
            t = matrixRot(alpha,beta,gama).I.dot(t).A1


            x, y, z = list(t)
            
            transLibViso[0].append(x)
            transLibViso[1].append(y)
            transLibViso[2].append(z)
            mx = min(z,mx)
            nx = max(z,nx)
            my = min(x,my)
            ny = max(x,ny)

        ra = max(nx-mx, ny-my)
        mx , nx = mx + (nx-mx)/2. - ra/2. - 0.1*ra, mx + (nx-mx)/2. + ra/2. + 0.1*ra
        my , ny = my + (ny-my)/2. - ra/2. - 0.1*ra,my + (ny-my)/2. + ra/2.  + 0.1*ra

        i = 1
        print path, step
        img = imread("features/{0}/step_{1}/I1_{2:06d}.png".format(path,step,i))
        i += step

        f = plt.figure()
        gs = GridSpec(1,2)
        #ax1 = plt.subplot(gs[0,1:])
        ax2 = plt.subplot(gs[0,1:])
        ax3 = plt.subplot(gs[0,0])
        #ax4 = plt.subplot(gs[1,0])
        #f, ((ax1,ax2),(ax3,ax4)) = subplots(2,2)
        ax2.imshow(img)

        ax3.clear()
        ax3.plot(transLibViso[2],transLibViso[0],"b")
        ax3.set_xlabel("X Axis (m)")
        ax3.set_ylabel("Y Axis (m)")
        ax3.set_xlim(mx,nx)
        ax3.set_ylim(my,ny)

        plt.show()


