#-*- coding: utf8 -*-
from cv2 import *
import os, rospy, csv
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from threading import Thread
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion, quaternion_from_euler

pathSave = os.path.expanduser("~/Dissertacao/datasets/car_simulation/carro_stereo_2014-06-12-02-14-53_color/")
frameLeft = frameRight = 1
bridge = CvBridge()
imgInfo = None

if not os.path.exists(pathSave):
    os.makedirs(pathSave)


def spin():
    rospy.spin()


def receiveImageLeft(img):
    global pathSave, frameLeft, bridge
    img = bridge.imgmsg_to_cv2(img)
#    img = cvtColor(img,cv.CV_BGR2GRAY)
    imwrite(pathSave+"I1_{0:06}.png".format(frameLeft),img)
    frameLeft += 1


def receiveImageRight(img):
    global pathSave, frameRight, bridge
    img = bridge.imgmsg_to_cv2(img)
#    img = cvtColor(img,cv.CV_BGR2GRAY)
    imwrite(pathSave+"I2_{0:06}.png".format(frameRight),img)
    frameRight += 1


def receiveImageInfo(info):
    global imgInfo
    imgInfo = info


def receivePosition(p):
    roll, pitch, yaw = euler_from_quaternion([p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w])
    pose = [p.pose.position.x, p.pose.position.y, p.pose.position.z, roll, pitch, yaw]
    print pose


if __name__ == "__main__":
    rospy.init_node("dissertacao")
    Thread(target = spin).start()
    rospy.Subscriber("/cat/pose", PoseStamped, receivePosition)
    rospy.Subscriber("/cat/semanticR/image", Image, receiveImageRight)
    rospy.Subscriber("/cat/semanticL/image", Image, receiveImageLeft)
    rospy.Subscriber("/cat/semanticL/camera_info", CameraInfo, receiveImageInfo)

