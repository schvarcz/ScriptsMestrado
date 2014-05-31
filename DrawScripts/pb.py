# -*- coding: utf8 -*-
from cv2 import *

path = "../nao/naooo_2014-03-10-17-48-35/"
pathsalvar = "../nao/naooo_2014-03-10-17-48-35_gray/"

for frame in range(285):
    img = imread("{0}I1_{1:06d}.png".format(path,frame))

    imgg = cvtColor(img,cv.CV_RGB2GRAY)

    imwrite("{0}I1_{1:06d}.png".format(pathsalvar,frame),imgg)
