from cv2 import *
import numpy as np

def mouse(event,x,y,a,b):
    if (event == EVENT_LBUTTONDOWN):
        print x,y,a,b
imgs = [
        ["chess.jpg",(13,12)],
        ["left01.jpg",(9,6)],
        ["calib3d.png",(6,8)],
    ]
index = 2
ch = imread(imgs[index][0])

#Numero de linhas e colunas respectivamente
chessSize = imgs[index][1]

found,points = findChessboardCorners(ch,chessSize)
if found:

	ptsW = [[3.*i, 3.*j, 0.] for j in range(imgs[index][1][1]) for i in range(imgs[index][1][0])]

	print len(points), len(ptsW)
	points = points.reshape((len(points),2))
	print points.shape

	A = []
	for i in range(len(ptsW)):
		print i 
		A.append([ptsW[i][0],ptsW[i][1], 1,0,0,0, -points[i][0]*ptsW[i][0], -points[i][0]*ptsW[i][1]])
		A.append([0,0,0, ptsW[i][0],ptsW[i][1], 1 , -points[i][1]*ptsW[i][0], -points[i][1]*ptsW[i][1]])
		#A.append([ptsW[i][0],ptsW[i][1],ptsW[i][2], 1, 0,0,0,0, -points[i][0]*ptsW[i][0], -points[i][0]*ptsW[i][1], -points[i][0]*ptsW[i][2]])
		#A.append([0,0,0,0, ptsW[i][0],ptsW[i][1],ptsW[i][2], 1 , -points[i][1]*ptsW[i][0], -points[i][1]*ptsW[i][1], -points[i][1]*ptsW[i][2]])

	A = np.matrix(A)

	b = np.matrix(points.reshape(len(points)*2)).T

	print np.linalg.solve(A.T*A,A.T*b)
drawChessboardCorners(ch,chessSize,points,found)

imshow("Chessboard",ch)
setMouseCallback("Chessboard",mouse)
waitKey()
