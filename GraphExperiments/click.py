from numpy import deg2rad, rad2deg
from math import sin,cos, atan2,sqrt
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.random.rand(10))
pts = [[0,0, 0]]
def onclick(event):
    global pts, ax
    
    ang = rad2deg(atan2(event.ydata - pts[-1][1],event.xdata - pts[-1][0]))
    if ang< 0:
        ang += 360
#    print '[%f, %f],'%(event.xdata, event.ydata)
    print '[%f, %f, %f],'%(event.xdata, event.ydata, ang)
    
    pts.append([event.xdata, event.ydata,ang])
    ax.clear()
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    x,y,angs = zip(*pts)
    ax.plot(x,y)
    fig.show()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
fig.show()
plt.show()
