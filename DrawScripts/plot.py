    # -*- coding: utf8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm, imread
from matplotlib.gridspec import GridSpec
import numpy as np
import csv, os
from numpy import *
from math import atan2


def matrixRot(alpha,beta,gama):
    return matrix( 
    [ [+cos(beta)*cos(gama), -cos(beta)*sin(gama), +sin(beta)],
    [+sin(alpha)*sin(beta)*cos(gama)+cos(alpha)*sin(gama),-sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama),-sin(alpha)*cos(beta)],
    [-cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama), +cos(alpha)*sin(beta)*sin(gama)+sin(alpha)*cos(gama), +cos(alpha)*cos(beta)]]) 


plt.ion()
f = plt.figure()

paths = [ 
#            "2010_03_09_drive_0019",
#            "drone/20140318_132620",
#            "drone/20140318_132620_gray",
#            "drone/20140318_133931",
#            "drone/20140318_133931_gray",
            "drone/20140327_135316_gray",
            "drone/20140328_102444_gray",
#            "nao/nao2",
#            "nao/nao2_gray",
#            "nao/nao2_rect",
#            "nao/nao2_rect_escolha",
#            "nao/naooo_2014-03-10-17-48-35",
#            "nao/naooo_2014-03-10-17-48-35_gray"
            ]

steps = [1]#,2,3,5,10,15,20]

pathResultados = os.path.expanduser("~/Dissertacao/OdometriaVisual/")
for skip in [1]:
    for path in paths:
        pathFull = os.path.expanduser("~/Dissertacao/datasets/"+path)
        for step in steps:
            transLibViso = [[0.],[0.],[0.]]
            oldR = asarray([.0,0.,0.]) #rotação da camera em relação ao gps
#            oldR = asarray([0.1,0.,0.5]) #rotação da camera em relação ao gps
            mx = my = float("inf")
            nx = ny = float("-inf")
            angles = [0.]
            for line in csv.reader(file(pathFull+"/posicoes_{}.csv".format(step)),delimiter=','):
                t = [float(l) for l in line]

                alpha, beta, gama = oldR
                t = matrixRot(alpha,beta,gama).I.dot(t).A1


                x, y, z = list(t)
                angles.append(angles[-1]+np.rad2deg(atan2(z-transLibViso[2][-1],x-transLibViso[2][-1])))
                
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

            pathSalvar = pathResultados+"/figures/{0}/step_{1}/skip_{2}".format(path,step,skip)
            if(not os.path.exists(pathSalvar)):
                os.makedirs(pathSalvar)

            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
            for j in xrange(0,len(transLibViso[0]),skip):
                f.clear()
                gs = GridSpec(1,2)
                #ax1 = plt.subplot(gs[0,1:])
                ax2 = plt.subplot(gs[:,1])
                ax3 = plt.subplot(gs[0:,0])
                #ax4 = plt.subplot(gs[-1,1])
                #f, ((ax1,ax2),(ax3,ax4)) = subplots(2,2)
                try:
                    img = imread(pathResultados+"/features/{0}/step_{1}/I1_{2:06d}.png".format(path,step,j))
                    ax2.imshow(img)
                    ax3.clear()
                    ax3.plot(transLibViso[0][0:j],transLibViso[2][0:j],"b")
                    ax3.set_xlabel("X Axis (m)")
                    ax3.set_ylabel("Y Axis (m)")
#                    ax3.set_xlim(-10,80)
#                    ax3.set_ylim(-20,140)
#                    ax3.set_xlim(mx,nx)
#                    ax3.set_ylim(my,ny)
                    ax3.set_ylim(mx,nx)
                    ax3.set_xlim(my,ny)


#                    ax4.clear()
#                    ax4.plot(angles[:j])
#                    ax4.set_xlabel("Frame")
#                    ax4.set_ylabel("Angle")
#                    ax4.set_xlim(0,len(transLibViso[0]))
#                    ax4.set_ylim(-180,180)
                    f.canvas.draw()
                    f.savefig("{0}/fig_{1:06d}.png".format(pathSalvar,j))
                    f.canvas.get_tk_widget().update() 
                    del img
                except:
                    pass

