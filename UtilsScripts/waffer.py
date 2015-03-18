#!/usr/bin/python
# -*- coding: utf8 -*-
from cv2 import *
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

def showHelp():
    print """
Convert to grayscale, equalize by patch

\t$ waffer folderToProcess [folderToSave]

          """
    sys.exit()

if len(sys.argv) < 2:
    showHelp()


for opt in sys.argv:
    if opt in ["-h","--help"]:
        showHelp()

pathSave = pathOrigin = os.path.expanduser(sys.argv[1])

if len(sys.argv) > 2:
    pathSave = os.path.expanduser(sys.argv[2])

if not os.path.exists(pathOrigin):
    print "The origin folder doesn't exist!!!"
    print "\tOrigin folder: ", pathOrigin
    sys.exit()

if not os.path.exists(pathSave):
    os.mkdir(pathSave)

print "Origin: ", pathOrigin
print "Save path: ", pathSave



def wafer(img, size):
    maxY, maxX = img.shape

    for y in range(0,maxY,size[0]):
        for x in range(0,maxX,size[1]):
            rect = img[y:y+size[0],x:x+size[1]]
            rect = (rect-rect.mean())/(3*rect.std())
            img[y:y+size[0],x:x+size[1]] = rect
    return img
size = (120,120)
    
for f in os.listdir(pathOrigin):
    print f
    img = cvtColor(imread(pathOrigin+"/"+f).astype("float32"),cv.CV_RGB2GRAY)
    img = 255*wafer(img,size)
    imwrite(pathSave+"/"+f,img)
    imshow("img",img)
    waitKey(33)
