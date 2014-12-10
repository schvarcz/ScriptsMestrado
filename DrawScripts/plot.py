# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm, imread
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv, os, sys
from numpy import *
from math import atan2

delay = "--delay" in sys.argv
if delay:
    plt.ion()
f = plt.figure()

paths = [
#            "2010_03_09_drive_0019",
#            "drone/20140318_132620",
#            "drone/20140318_132620_gray",
#            "drone/20140318_133931",
#            "drone/20140318_133931_gray",
#            "drone/20140327_135316_gray",
#            "drone/20140328_102444_gray",
             "motox/VID_20140617_162058756_ESCOLHA",
#             "motox/2014-11-04_ESCOLHA",
#             "motox/VID_20140617_163505406_GRAY",
#             "motox/VID_20140617_162058756_GRAY",
#             "motox/VID_20140617_162058756_GRAY_ESCOLHA",
#             "motox/VID_20140617_162058756_GRAY_equalized",
#             "motox/VID_20140617_162058756_GRAY_CURVA",
#             "motox/VID_20140617_162058756_GRAY_small_CURVA",
#             "motox/VID_20140617_162058756_GRAY_equalized_small_CURVA",
#             "motox/VID_20140617_162058756_GRAY_equalized_small",
#             "motox/VID_20140617_162058756_GRAY_equalized_ESCOLHA",
#             "motox/VID_20140617_162058756_GRAY_equalized_small_ESCOLHA2",
#             "motox/VID_20140617_162058756_GRAY_equalized_small_GRAMA_ESCOLHA",
#             "motox/VID_20140617_162058756_GRAY_ESCOLHA_GRAMA",
#            "nao/nao2",
#            "nao/nao2_gray",
#            "nao/nao2_rect",
#            "nao/nao2_rect_escolha",
#            "nao/naooo_2014-03-10-17-48-35",
#            "nao/naooo_2014-03-10-17-48-35_gray"
            ]

steps = [1]#,2,3,5,10,15,20]

pathResultados = os.path.expanduser("~/Dissertacao/OdometriaVisual/")
matchers = [
#        "",
#        "_eq_match_sift_constrainted",
        "_eq_match_sift",
#        "_match_sift",
#        "_match_surf",
#        "_match_surf_sift",
#        "_knn_sift",
#        "_knn_surf",
#        "_knn_surf_sift",
#        "_knn_sift_closer",
#        "_knn_surf_closer",
#        "_knn_surf_sift_closer"
        ]
for matcher in matchers:
    for skip in [1]:
        for path in paths:
            for step in steps:
                transLibViso = [[0.],[0.],[0.]]
                rotLibViso = [[0.],[0.],[0.]]
                mx = my = mz = float("inf")
                nx = ny = nz = float("-inf")
                for line in csv.reader(file(pathResultados+"/features/{0}/step{3}_{1}/posicoes_{2}.csv".format(path,step,step,matcher)),delimiter=','):
                    t = [float(l) for l in line[:-1]]
                    t,r = t[:3],t[3:]

                    alpha, gama, beta  = r
                    x, z, y = t

                    transLibViso[0].append(x)
                    transLibViso[1].append(y)
                    transLibViso[2].append(z)

                    rotLibViso[0].append(rad2deg(alpha))
                    rotLibViso[1].append(rad2deg(beta))
                    rotLibViso[2].append(rad2deg(gama))
                    mx = min(x,mx)
                    nx = max(x,nx)
                    my = min(y,my)
                    ny = max(y,ny)
                    mz = min(z,mz)
                    nz = max(z,nz)

                ra = max(nx-mx, ny-my)
                mx , nx = mx + (nx-mx)/2. - ra/2. - 0.1*ra, mx + (nx-mx)/2. + ra/2. + 0.1*ra
                my , ny = my + (ny-my)/2. - ra/2. - 0.1*ra,my + (ny-my)/2. + ra/2.  + 0.1*ra

                pathSalvar = pathResultados+"/figures/{0}/step{3}_{1}/skip_{2}".format(path,step,skip,matcher)
                if(not os.path.exists(pathSalvar)):
                    os.makedirs(pathSalvar)

                plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
                skips = [len(transLibViso[0])-2]
                if delay:
                    skips = xrange(0,len(transLibViso[0]),skip)
                for j in skips:
                    f.clear()
                    gs = GridSpec(3,2)
                    #ax1 = plt.subplot(gs[0,1:])
                    ax2 = plt.subplot(gs[:-1,1])
                    ax3 = plt.subplot(gs[0:,0]) #, projection='3d')
                    ax4 = plt.subplot(gs[-1,1])
                    #f, ((ax1,ax2),(ax3,ax4)) = subplots(2,2)
                    try:
                        img = imread(pathResultados+"/features/{0}/step{3}_{1}/I1_{2:06d}.png".format(path,step,j,matcher))
                        ax2.imshow(img)
                        ax2.set_title("Frame {0}".format(j))
                        ax3.clear()
                        ax3.plot(transLibViso[0][:j],transLibViso[1][:j])#,transLibViso[2][:j],"b")
                        ax3.set_xlabel("X Axis (m)")
                        ax3.set_ylabel("Y Axis (m)")
#                        ax3.set_xlim(-10,80)
#                        ax3.set_ylim(-20,140)
                        ax3.set_xlim(mx,nx)
                        ax3.set_ylim(my,ny)
                        #ax3.set_zlim(mz,nz)


                        ax4.clear()
                        ax4.plot(rotLibViso[2][:j])
                        ax4.set_xlabel("Frame")
                        ax4.set_ylabel("Angle")
                        ax4.set_xlim(0,len(transLibViso[2]))
#                        ax4.set_ylim(-180,180)
                        if delay:
                            f.canvas.draw()
                            f.savefig("{0}/fig_{1:06d}.png".format(pathSalvar,j))
                            f.canvas.get_tk_widget().update()
                        else:
                            plt.show()
                        del img
                    except:
                        pass

