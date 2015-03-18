#!/usr/bin/python
from matplotlib import pyplot as plt
from numpy import *
import cv2
from math import *
from numpy.linalg import solve
from numpy.random import random_sample


f1 = plt.figure()
f1.show()
maxWindowSize = 5

imgGlobal = cv2.imread("results.bmp")
figIdx = 0

def show(img,line, hist = None, histTitle = None):
    f1.clear()
    if hist != None:
        ax1 = f1.add_subplot(222)
    else:
        ax1 = f1.add_subplot(122)
    ax1.imshow(img)
    #plt.colorbar()
    ax1.set_xlim(0,29+2*maxWindowSize)
    ax1.set_ylim(29+2*maxWindowSize,0)

    x,y = zip(*line)
    ax1.plot(x,y,"b")
    ax1.plot(x,y,"+r")

    if hist != None:
        ax2 = f1.add_subplot(221)
    else:
        ax2 = f1.add_subplot(121)
    
    ax2.imshow(imgGlobal)
    
    if hist != None:
        ax3 = f1.add_subplot(212)
        ax3.plot(hist)
        ax3.set_ylim(0,9)
        if histTitle != None:
            ax3.set_title(histTitle)
        

    #plt.show()
    f1.canvas.draw()
    f1.canvas.flush_events()
    
    global figIdx
    f1.savefig("imgs/I_{0:06}.png".format(figIdx))
    figIdx += 1
    #cv2.waitKey(33)


def relaxLineMeanShift(img,line):
    maxY, maxX = img.shape
    for idx in range(len(line)):
        pt = line[idx]
        x,y = pt
        while True:
            window = slice(max(x-7,0),min(x+8,maxX))
            show(rectVB,line,img[y,window])
            newMeanshift = (  arange(window.start,window.stop) * img[y,window]  ).sum()/img[y,window].sum()
            #print y, x ,newMeanshift
            if round(newMeanshift) == x:
                break
            x = round(newMeanshift)
            line[idx] = [x,y]
    show(rectVB,line)


def relaxLineMeanShift2(img,line):
    maxY, maxX = img.shape
    for idx in range(len(line)):
        pt = line[idx]
        x,y = pt
        while True:
            windowSizeX = min(x-max(x-maxWindowSize,0), min(x+maxWindowSize,maxX-1)-x)
            windowSizeY = min(y-max(y-maxWindowSize,0), min(y+maxWindowSize,maxY-1)-y)

            windowX = slice(x-windowSizeX,x+windowSizeX+1)
            windowY = slice(y-windowSizeY,y+windowSizeY+1)

            weights = asarray([img[windowY, nX].sum() for nX in range(int(windowX.start), int(windowX.stop))])

            newMeanshift = (  arange(windowX.start,windowX.stop) * weights  ).sum()/weights.sum()

            print y, x, windowX.start, windowX.stop, newMeanshift-windowX.start
            show(rectVB,line,weights)
            desv = newMeanshift-x

            if abs(desv)<1 and abs(desv) > 0.1:
                if desv < 0:
                    desv = -1
                if desv > 0:
                    desv = 1

            newX = round(desv+x)
            if newX == x:
                break
            x = newX
            line[idx] = [x,y]
    show(rectVB,line)


def relaxLineMeanShift3(img,line):
    maxY, maxX = img.shape
    for idx in range(len(line)):
        pt = line[idx]
        x,y = pt
        oldX = None
        while True:
            windowSizeX = min(maxWindowSize-max(maxWindowSize - x - 1,0), min(x+maxWindowSize,maxX-1)-x)
            windowSizeY = min(maxWindowSize-max(maxWindowSize - y - 1,0), min(y+maxWindowSize,maxY-1)-y)

            windowX = slice(x-windowSizeX+1,x+windowSizeX)
            #print windowX.start, " - ", windowX.stop
            windowY = slice(y-windowSizeY+1,y+windowSizeY)
            #print windowY.start, " - ", windowY.stop

            weights = asarray([img[windowY, nX].sum() for nX in range(int(windowX.start), int(windowX.stop))])
            #print weights

            newMeanshift = (  arange(windowX.start,windowX.stop) * weights  ).sum()/weights.sum()

            #print x, y, windowX.start, windowX.stop, newMeanshift-windowX.start
            #show(rectVB,line,weights, histTitle = "Node {0}".format(idx))
            #print newMeanshift
            desv = newMeanshift-x

            if 0.1 < abs(desv) < 1:
                if desv < 0:
                    desv = -1
                if desv > 0:
                    desv = 1

            newX = round(desv+x)
            if newX == oldX:
                if abs(oldDesv) > abs(desv):
                    x = newX
                    moveLine(line, idx, desv)
                break
            elif newX == x:
                break
            
            oldX, x, oldDesv = x, newX, desv
            moveLine(line, idx, desv)
            
#        for i in range(10):
#            show(rectVB,line,weights, histTitle = "Node {0}".format(idx))
#    show(rectVB,line)


def moveLine(line, idx, desv):
    for uIdx in range(idx,len(line)):
        uPt = line[uIdx]
        uX,uY = uPt
        newX = round(desv+uX)
        if newX == uX:
            break
        uX = newX
        line[uIdx] = [uX,uY]


def relaxLine(img,line):
    erro = 0
    maxY, maxX = img.shape
    while True:
        for idx in range(len(line)):
            pt = line[idx]
            x,y = pt
            move = 0
            ptAnterior = ptPosterior = None
            if idx != 0 and idx != len(line)-1:
                ptAnterior = line[idx-1]
                move += ptAnterior[0] - x

                ptPosterior = line[idx+1]
                move += ptPosterior[0] - x
            dx, dy = (img[y,x+1] - img[y,x-1])/2. , (img[y+1,x] - img[y-1,x])/2.
            th = atan2(-dy,-dx)
            x, y = x+cos(th), y#+sin(th)
            #x -= dx
#            if abs(move) != 0:
#                move = move/abs(move)
#                x += move
            print move
            x = min(max(x,0),maxX-2)
            y = min(max(y,0),maxY-2)
            line[idx] = [x, y]
        show(rectVB,line)


def ransac(pts, model, iterations=1000, sampleSize=3, threshold = 5, minInliers = 28):
    maxInliers = 0
    bestParams, bestInliers = None, None
    ptsA = asarray(pts)
    for i in range(iterations):
        random_selection = (random.random_sample(3)*len(pts)).astype(int)
        params = model.model(ptsA[random_selection].tolist())
        if params == None:
            continue
        inliers, outliers = model.fit(pts, params, threshold)
        
        if len(inliers) > maxInliers:
            maxInliers = len(inliers)
            bestInliers = inliers
            bestParams = params
        
        if maxInliers >= minInliers:
            break
    
    params = model.model(bestInliers)
    inliers, outliers = model.fit(pts, params, threshold)
    return params, inliers, outliers


class LSE(object):

    def model(self, pts):
        x, y = zip(*pts)
        x, y = asarray(x), asarray(y)
        A = matrix([x**3,x**2,x,ones(x.shape)]).T
        b = matrix(y).T
        try:
            params = solve(A.T*A,A.T*b).A1
            #print "A = {0} B = {1} C = {2} D = {3}".format(params[0], params[1], params[2], params[3])
            return params
        except:
            return None
    
    
    def fit(self, pts, params, threshold):
        x, y = zip(*pts)
        x, y = asarray(x), asarray(y)
        erros = y - (params[0]*x**3+params[1]*x**2+params[2]*x+params[3])
        inliers, outliers = [], []
        for idx in range(len(erros)):
            erro = erros[idx]
            if abs(erro) < threshold:
                inliers.append(pts[idx])
            else:
                outliers.append(pts[idx])
        #print "erros: ", erros
        return inliers, outliers
    
    
    def fixLine(self, pts, params):
        x, y = zip(*pts)
        x, y = asarray(x), asarray(y)
        y = params[0]*x**3+params[1]*x**2+params[2]*x+params[3]
        return zip(x,y)


y ,x = 145, 15
rectV = cv2.cvtColor(imgGlobal[y-maxWindowSize:y+30+maxWindowSize,x-maxWindowSize:x+30+maxWindowSize],cv2.cv.CV_RGB2GRAY).astype(float32)
#plt.imshow(rectV)
#plt.show()
ma, mi = rectV.max(), rectV.min()
rectV = (rectV-mi)/(ma-mi)
rectVB = rectV #cv2.blur(cv2.dilate(cv2.blur(rectV,(3,3)),ones([3,3])),(3,3))

maxY, maxX = rectVB.shape
rectVB = -rectVB+1.

#plt.imshow(rectVB)
#plt.colorbar()
#plt.show()

line = zip(*[arange(0+maxWindowSize,29+maxWindowSize),arange(0+maxWindowSize,29+maxWindowSize)])

plt.ion()

relaxLineMeanShift3(rectVB,line)
print line

model = LSE()
params, inliers, outliers = ransac(line, model)
lineFixed = model.fixLine(line,params)

plt.ioff()

f1.clear()

ax1 = f1.add_subplot(111)
ax1.imshow(rectVB)
ax1.set_xlim(0,29+2*maxWindowSize)
ax1.set_ylim(29+2*maxWindowSize,0)

x,y = zip(*line)
ax1.plot(x,y,"b")
ax1.plot(x,y,"+r")

x, y = zip(*lineFixed)
ax1.plot(x, y, 'g')
x, y = zip(*inliers)
ax1.plot(x, y, 'b+')

if len(outliers) > 0:
    x, y = zip(*outliers)
    ax1.plot(x, y, 'r+')
plt.show()
