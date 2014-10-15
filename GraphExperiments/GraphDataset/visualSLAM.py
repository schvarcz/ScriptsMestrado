import numpy as np
from math import sin,cos
import cv2

def ssd(template,comp):
    sd = np.power(comp-template,2)
    return sd.sum()

def rotMatrix3dZYX(yaw, pitch,roll):
    return np.matrix([
        [cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - cos(roll)*sin(yaw), sin(yaw)*sin(roll)+ cos(yaw)*cos(roll)*sin(pitch)],
        [cos(pitch)*sin(yaw),  cos(yaw)*cos(roll)+sin(yaw)*sin(pitch)*sin(roll), cos(roll)*sin(yaw)*sin(pitch)-cos(yaw)*sin(roll)],
        [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)],
        ])

def rotMatrix2dZYX(yaw, pitch,roll):
    return np.matrix([
        [cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - cos(roll)*sin(yaw), sin(yaw)*sin(roll)+ cos(yaw)*cos(roll)*sin(pitch)],
        [cos(pitch)*sin(yaw),  cos(yaw)*cos(roll)+sin(yaw)*sin(pitch)*sin(roll), cos(roll)*sin(yaw)*sin(pitch)-cos(yaw)*sin(roll)],
        [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)],
        ])
        
def rotMatrix2d(angle):
    return np.matrix([
                [cos(angle), -sin(angle), 0.],
                [sin(angle), cos(angle), 0.],
                [0., 0., 1.],
            ])
class Node(object):
    nodes = 1

    def __init__(self,pose,imgName):
        self.nodeid = Node.nodes
        self.pose = np.asarray(pose)
        self.imgName = imgName
        self.aresta_from = []
        self.aresta_to = []
        Node.nodes += 1

    def tolist(self):
        return self.pose.tolist()

    def img(self):
        return cv2.imread(self.imgName)

    def __repr__(self):
        return "ID: {0}\nPose: {1}".format(self.nodeid,self.pose)

class Graph(object):
    def __init__(self):
        self.nodes = []

    def relaxMilford(self):
        delta_total = np.zeros(6)
        for pt in self.nodes:
            sum_from = np.zeros(6)
            for f,transition in pt.aresta_from:
                sum_from += pt.pose - f.pose - transition

            sum_to = np.zeros(6)
            for t,transition in pt.aresta_to:
                sum_to += t.pose - pt.pose - transition
            delta = .5*(sum_to - sum_from)
            if pt.nodeid == 7:
                print "Position: ", pt.pose
                print "aresta_from: ", pt.aresta_from
                print "aresta_to: ", pt.aresta_to
                print "DELTA: ", delta, sum_to, sum_from
            pt.pose += delta
            roll = -np.deg2rad(delta[3])
            pitch = -np.deg2rad(delta[4])
            yaw = -np.deg2rad(delta[5])

            #####################
            # Update transition #
            #####################
            for i in range(len(pt.aresta_to)):
                t, transition = pt.aresta_to[i]

                rot = np.matrix(np.eye(6,6))
                rot[:3,:3] = rotMatrix3dZYX(yaw,pitch,roll)
                transition_new = (rot * np.matrix(transition).T).T.A.reshape((6,))
                #transition_new = (rotMatrix2d(angle) * np.matrix(transition).T).T.A.reshape((3,))
                pt.aresta_to[i][1] = transition_new
                node_to = pt.aresta_to[i][0]

                for index in range(len(node_to.aresta_from)):
                    if node_to.aresta_from[index][0] == pt and (node_to.aresta_from[index][1] == transition).all():
                        break
                #index = node_to.aresta_from.index([pt, transition])
                node_to.aresta_from[index][1] = transition_new
            delta_total += delta
        return abs(delta_total)

    def relaxMine(self):
        pass

    def revisited(self,newNode):
        ret = float("inf"), ""
        for n in self.nodes[0:55]:
            result = ssd(newNode.img()[100:250,400:1000],n.img()[100:250,400:1000])
            if ret[0] > result:
                ret = result, n
        return ret

    def tolist(self):
        return [n.tolist() for n in self.nodes]

    def append(self, node):
        self.nodes.append(node)

    def __getitem__(self,index):
        return self.nodes[index]

    def __setitem__(self, index, item):
        self.nodes[index] = item

