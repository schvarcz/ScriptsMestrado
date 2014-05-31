#-*- coding: utf8 -*-
from cv2 import *
from math import sqrt
import numpy as np, rospy
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image as PImage
from geometry_msgs.msg import PoseStamped
from threading import Thread

pubImg = None
pubInfo = None
imgInfo = None

def PIL2numpy(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 3)


def ROS2numpy(img):
    return PIL2numpy(ROS2PIL(img))
#    return np.array(img.data).reshape(img.width, img.height, 3)


def ROS2PIL(img):
    return PImage.fromstring("RGB",(img.width,img.height),img.data)


def numpy2ROS(npImg):
    msgImg = Image()
#    h, w, chs = npImg.shape
    h, w = npImg.shape
    msgImg.height = h
    msgImg.width = w
    msgImg.data = npImg.tostring()
#    msgImg.encoding = "bgr8"
#    msgImg.step = 1*w*chs
    msgImg.encoding = "mono8"
    msgImg.step = 1*w
    return msgImg


def spin():
    rospy.spin()


def receiveImage(img):
    print img.encoding
    img = ROS2numpy(img)
    img = cvtColor(img,cv.CV_BGR2GRAY)
    img = numpy2ROS(img)
    global imgInfo
    if imgInfo != None:
        print "publicando"
        pubImg.publish(img)
        pubInfo.publish(imgInfo)
    else:
        print "Sem camera info...."


def receiveImageInfo(info):
    global imgInfo
    imgInfo = info


def receivePosition(p):
    print p


if __name__ == "__main__":
    rospy.init_node("dissertacao")
    Thread(target = spin).start()
    pubImg = rospy.Publisher('/image', Image)
    pubInfo = rospy.Publisher('/camera_info', CameraInfo)
    rospy.Subscriber("/mono_odometer/pose", PoseStamped, receivePosition)
    rospy.Subscriber("/nao_camera/image_raw", Image, receiveImage)
    rospy.Subscriber("/nao_camera/camera_info", CameraInfo, receiveImageInfo)

