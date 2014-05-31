#-*- coding: utf8 -*-
from cv2 import *
from math import sqrt
import numpy as np, rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from threading import Thread
video = VideoCapture("ratslam_ufrgs_full.mp4")

k = 0

def spin():
    rospy.spin()

def receivePosition(p):
    print p

if __name__ == "__main__":
    rospy.init_node("dissertacao")
    Thread(target = spin).start()
    pubImg   = rospy.Publisher('/image', Image)
    rospy.Subscriber("/mono_odometer/pose", PoseStamped, receivePosition)
    while (video.grab() and k != 27):
        ret,img = video.retrieve()
#        imshow("Video",img)
        img = cvtColor(img,cv.CV_BGR2GRAY)
        msgImg = Image()
        h, w, chs = img.shape
        msgImg.height = h
        msgImg.width = w
        msgImg.data = img.tostring()
        msgImg.encoding = "bgr8"
#        msgImg.encoding = "mono8"
        msgImg.step = 1*w*chs
        pubImg.publish(msgImg)
        k = waitKey(20) & 255

    destroyAllWindows()

