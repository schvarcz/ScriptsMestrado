# -*- coding:utf8 -*-
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cv2 import *
import os

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

steps = [2,3,5,10,15,20,25]

for path in paths:
    for step in steps:
        f = open(path+"/features_{}.csv".format(step))
        pathSalvar = "features/"+path+"/step_{}".format(step)
        if(not os.path.exists(pathSalvar)):
            os.makedirs(pathSalvar)
        frame = 0
        img = imread("{0}/I1_{1:06d}.png".format(path,frame))
        for l in f.readlines():
            if (l[:6] == "imagem"):
                imwrite(pathSalvar+"/I1_{0:06d}.png".format(frame),img)
                frame += step
                img = imread("{0}/I1_{1:06d}.png".format(path,frame))
                print "Frame ",frame
            else:
                #print l
                prev, cur = l.split(";")
                up,vp = prev.split(",")
                uc,vc = cur.split(",")
                up, vp, uc, vc = int(up), int(vp), int(uc), int(vc)
                
                line(img,(up, vp),(uc, vc),(255,0,0))
                circle(img,(uc, vc),3,(0,255,0),-1)
        f.close()

