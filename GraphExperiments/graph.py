from numpy import deg2rad
from math import sin,cos
import numpy as np
from matplotlib import pyplot as plt

movimentos = [
    [-6.198511, -3.469388, 209.236310],
    [-4.173697, 3.061224, 72.773970],
    [-0.044665, 5.765306, 33.220579],
    [4.600496, 4.540816, 345.232458],
    [5.593052, 0.969388, 285.531443],
    [5.751861, -3.469388, 272.049034],
    [4.521092, -6.275510, 246.317676],
    [2.059553, -7.551020, 207.392108],
    [-0.838710, -8.265306, 193.844813],
    [-2.744417, -7.857143, 167.911066],
    [-2.347395, -6.632653, 72.035496],
    [-0.481390, -6.377551, 7.784663],
    [2.456576, -5.663265, 13.664780],
    [3.131514, -3.469388, 72.899687],
    [1.066998, -1.734694, 139.961601],
    [-0.600496, -2.397959, 201.690772],
    [-2.307692, -4.489796, 230.781325],
    [-4.411911, -4.387755, 177.223705],
    [-5.007444, -3.877551, 139.412751],
]


xo,yo, angles = zip(*movimentos)


pts = [np.asarray(m) for m in movimentos]
transitions = [ pts[i+1]-pts[i] for i in range(len(pts)-1)]
transitions.append(pts[0]-pts[-1])
movimentos.remove(movimentos[-1])
movimentos.append(movimentos[0])
pts = [np.asarray(m) for m in movimentos]

for j in range(100):
    for i in range(len(pts)):
        next = i+1
        if next >= len(pts):
            next =  next - len(pts)
        delta = 0.5*((pts[next] - pts[i] - transitions[i]) + (pts[i] - pts[i-1] - transitions[i-1]))
        pts[i] += delta
        angle = deg2rad(pts[i][0])
#        print (np.matrix([
#            [1., 0., 0.],
#            [0., cos(angle), -sin(angle)],
#            [0., sin(angle), cos(angle)],
#        ])* np.matrix(transitions[i]).T).T.A.reshape((3,))
        transitions[i] = (np.matrix([
            [cos(angle), -sin(angle), 0.],
            [sin(angle), cos(angle), 0.],
            [0., 0., 1.],
        ]) * np.matrix(transitions[i]).T).T.A.reshape((3,))
        print delta.tolist()

    x,y, angles = zip(*[pt.tolist() for pt in pts])

    plt.plot(xo,yo,"g")
    plt.plot(x,y,"b")
#    plt.xlim(-10,10)
#    plt.ylim(-10,10)
    plt.show()
    
