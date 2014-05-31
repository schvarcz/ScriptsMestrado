#-*- coding: utf8 -*-
from cv2 import *
from math import sqrt
import numpy as np, rospy
from threading import Thread
from matplotlib import pyplot as plt
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from time import sleep
from mpl_toolkits.mplot3d import Axes3D

plt.ion()
f = plt.figure()
ax = f.add_subplot(111,projection='3d')

ptsVisual = []
ptsOdom = []

def spin():
    rospy.spin()


def receivePosition(p):
    global ax, ptsVisual, f
    ptsVisual.append([p.pose.position.x, p.pose.position.y, p.pose.position.z])


def receiveOdom(p):
    global ax, ptsOdom, f
    ptsOdom.append([p.pose.pose.position.x, p.pose.pose.position.y, p.pose.pose.position.z])


if __name__ == "__main__":
    rospy.init_node("dissertacao")
    Thread(target = spin).start()
    rospy.Subscriber("/mono_odometer/pose", PoseStamped, receivePosition)
    rospy.Subscriber("/odom", Odometry, receiveOdom)

    while(True):
        ax.clear()
        if len(ptsVisual) > 0:
            x, y, z = zip(*ptsVisual)
            ax.plot(x,y,z)
        if len(ptsOdom) > 0:
            x, y, z = zip(*ptsOdom)
            ax.plot(x,y,z)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
#        ax.set_xlim3d(-2,2)
#        ax.set_ylim3d(-2,2)
#        ax.set_zlim3d(-2,2)
        f.canvas.get_tk_widget().update() 
        f.canvas.draw()
    

