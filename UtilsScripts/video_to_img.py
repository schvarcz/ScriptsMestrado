# -*- coding: utf8 -*-
from cv2 import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm
from matplotlib.gridspec import GridSpec
import numpy as np, os


path = os.path.expanduser("~/Dissertacao/datasets/motox/VID_20140617_162058756")
video = VideoCapture(path+".mp4")
path = path + "_GRAY"

frame = 0

if not os.path.exists(path):
    os.mkdir(path)

while (video.grab()):
    print "Frame ", frame
    b, img = video.retrieve()
    img = cvtColor(img,cv.CV_RGB2GRAY)
    imwrite("{0}/I1_{1:06d}.png".format(path,frame),img)
    frame += 1

