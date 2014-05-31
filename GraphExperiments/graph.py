from numpy import deg2rad
from math import sin,cos
import numpy as np
from matplotlib import pyplot as plt

movimentos = [
    [5.831266, 0.000000],
    [2.337086, 79.229989],
    [1.889424, 8.361404],
    [2.694801, 55.118362],
    [3.000664, 61.374083],
    [5.039702, 3.017537],
    [4.471391, 40.453140],
    [6.362794, 76.048748],
    [3.244465, -279.821277],

]


pts = [[0,0]]

pos = [0,0]
ang = deg2rad(0)
for mov in movimentos:
    ang += deg2rad(mov[1])
    pts.append([pts[-1][0]+cos(ang)*mov[0], pts[-1][1]+sin(ang)*mov[0]])
    
x, y = zip(*pts)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.plot(x,y)
plt.show()
    
