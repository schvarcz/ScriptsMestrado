from cv2 import *
import numpy as np
from matplotlib import pyplot as plt


def normalizePath(img,mode):
    if mode == "equalize":
        return equalizeHist(img)

    if mode == "minmax":
        mi, ma = img.min(), img.max()
        if mi ==0 or ma == 0 or mi == ma:
            print "Erro: ",img.min(), img.max()
            return img
        step = 255./(ma-mi)
        return step*(img-mi)

    if mode == "stdDesv":
        img = img.astype(np.float32)
        me, stdD = meanStdDev(img)
        if stdD == 0:
            stdD = 1.
        img = 127.+127.*(img-me)/float(stdD)
        img[img<0] = 0.
        img[img>255] = 255.
        return img.astype(np.uint8)

    return img


def patchlize(img):

    w,h = img.shape
    wWindow,hWindow = 18,32

    for j in xrange(h/hWindow):
        for i in xrange(w/wWindow):
            lowh, highh, loww, highw = hWindow*j, hWindow*(j+1), wWindow*i, wWindow*(i+1)
            img[loww:highw, lowh:highh] = normalizePath(img[loww:highw, lowh:highh],"stdDesv")
    return img

img = imread("I1_000000.png")

img = cvtColor(img,cv.CV_RGB2GRAY)
w,h = img.shape
img = resize(img,(h/5,w/5))

img = patchlize(img)

plt.imshow(img)
plt.colorbar()
plt.show()
imwrite("stdDesv.png",img)
