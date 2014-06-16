# -*- coding: utf8 -*-
from cv2 import *
import os, rospy
from sensor_msgs.msg import Image, CameraInfo
from threading import Thread
from cv_bridge import CvBridge

pathSalvar = os.path.expanduser("~/Dissertacao/datasets/nao/nao2_rect/")
frame = 0
b = CvBridge()

if not os.path.exists(pathSalvar):
    os.mkdir(pathSalvar)


def spin():
    rospy.spin()

def receiveImage(img):
    global frame, pathSalvar, b
    print "Recebeu Imagem!"
    img = b.imgmsg_to_cv2(img)

    if pathSalvar != "":
        imwrite("{0}I1_{1:06d}.png".format(pathSalvar,frame),img)
        frame += 1


def receiveCameraInfo(camera):
    print "Recebeu CameraInfo!"
    print camera


if __name__ == "__main__":
    rospy.init_node("MyVisualOdometryMono")

    rospy.Subscriber("/file_to_ros/image_rect", Image, receiveImage)
    rospy.Subscriber("/file_to_ros/camera_info", CameraInfo, receiveCameraInfo)
    Thread(target = spin).start()
