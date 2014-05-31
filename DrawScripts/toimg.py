# -*- coding: utf8 -*-
from cv2 import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm
from matplotlib.gridspec import GridSpec
import numpy as np, os


video = VideoCapture("rodrigo/MOV03936.MPG")

frame = 0
path = "rodrigo/guaiba/"

if not os.path.exists(path):
    os.mkdir(path)

while (video.grab()):
    print "Frame ", frame
    b, img = video.retrieve()
    img = cvtColor(img,cv.CV_RGB2GRAY)
    imwrite("{0}I1_{1:06d}.png".format(path,frame),img)
    frame += 1

