#-*- coding: utf8 -*-
from cv2 import *
from math import sqrt
import os, numpy as np, rospy
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image as PImage
from geometry_msgs.msg import PoseStamped
from threading import Thread
from time import sleep
from cv_bridge import CvBridge

bridge = CvBridge()

def spin():
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("dissertacao")
    pubImg = rospy.Publisher('/file_to_ros/image_raw', Image)
    pubInfo = rospy.Publisher('/file_to_ros/camera_info', CameraInfo)
    t = Thread(target = spin)
    t.start()
    raw_input()
    params = CameraInfo()
    params.header.frame_id = "/CameraTop"
    params.distortion_model = "plumb_bob"
    
    #Ardrone
    params.width = 640
    params.height = 360
    params.K = [ 564.030788, 0.000000, 320.583306,
                 0.000000, 566.501051, 164.804138,
                 0.000000, 0.000000, 1.000000]

    params.D = [-0.499559, 0.288443, -0.002316, 0.003517, 0.000000]

    params.R = [ 1.000000, 0.000000, 0.000000,
                 0.000000, 1.000000, 0.000000,
                 0.000000, 0.000000, 1.000000]

    params.P = [ 466.701813, 0.000000e+00, 324.0212752, 0.000000e+00,
                 0.000000e+00, 534.176575, 162.123830, 0.000000e+00,
                 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
    start, end = 1,17504
    path = os.path.expanduser("~/Dissertacao/datasets/drone/20140328_102444/")
    
    #NAO
    params.width = 640
    params.height = 480
    params.K = [ 553.861086, 0.0, 320.836884,
                 0.0, 553.642958, 244.827406,
                 0.0, 0.0, 1.0]

    params.D = [-0.074465, 0.092772, 0.000516, 0.000764, 0.0]

    params.R = [ 1.000000, 0.000000, 0.000000,
                 0.000000, 1.000000, 0.000000,
                 0.000000, 0.000000, 1.000000]

    params.P = [ 545.790894, 0.000000e+00, 320.824327, 0.000000e+00,
                 0.000000e+00, 546.458679, 244.557631, 0.000000e+00,
                 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
    start, end = 0, 761
    path = os.path.expanduser("~/Dissertacao/datasets/nao/nao2/")
    
    b = CvBridge()
    dtype, n_channels = b.encoding_as_cvtype2('8UC3')
    for i in range(start, end):
        img = imread(path+"I1_{:06}.png".format(i))
        print "Image ",i
#        imshow("teste",img)
        img = b.cv2_to_imgmsg(img)
        img.encoding = "bgr8"
        img.header.frame_id = "/CameraTop"
        params.header.stamp = img.header.stamp = rospy.get_rostime()
        pubImg.publish(img)
        pubInfo.publish(params)
#        waitKey(33)
#        sleep(0.3)
    rospy.signal_shutdown("")


