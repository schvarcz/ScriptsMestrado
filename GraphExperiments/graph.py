from numpy import deg2rad
from math import sin,cos
import numpy as np
from matplotlib import pyplot as plt

movimentos = [
    [-6.397022, 1.581633, 0.000000],
    [-4.451613, 3.061224, 37.255003],
    [-1.434243, 5.867347, 42.922506],
    [0.590571, 6.836735, 25.582955],
    [3.131514, 5.816327, 338.120276],
    [5.275434, 5.000000, 339.154997],
    [5.513648, 4.081633, 284.541363],
    [5.910670, 2.244898, 282.197206],
    [5.950372, 1.479592, 272.969704],
    [6.029777, -0.102041, 272.874071],
    [5.950372, -0.459184, 257.465171],
    [5.712159, -1.836735, 260.189135],
    [5.196030, -2.857143, 243.169419],
    [4.838710, -3.112245, 215.524233],
    [3.846154, -3.877551, 217.633873],
    [2.972705, -4.030612, 189.939460],
    [2.138958, -4.336735, 200.161454],
    [1.424318, -4.438776, 188.126123],
    [0.789082, -4.591837, 193.547289],
    [-0.679901, -4.438776, 174.051509],
    [-2.109181, -4.795918, 194.029556],
    [-3.419355, -4.795918, 180.000000],
    [-4.610422, -4.693878, 175.103326],
    [-5.285360, -4.642857, 175.677077],
    [-5.880893, -4.744898, 189.722845],
    [-6.357320, -4.693878, 173.887507],
    [-6.635236, -4.591837, 159.838546],
    [-7.032258, -4.387755, 152.795449],
    [-7.389578, -4.081633, 139.412751],
    [-7.627792, -3.520408, 112.998939],
    [-7.826303, -2.346939, 99.601599],
    [-7.905707, -1.836735, 98.846133],
    [-7.985112, -0.714286, 94.046486],
    [-7.905707, -0.459184, 72.710436],
    [-7.667494, 0.051020, 64.972211],
    [-7.588089, 0.153061, 52.111239],
    [-7.032258, 0.765306, 47.765020],
    [-6.476427, 0.714286, 50.]
]

#movimentos = [
#    [-6., 6., 0.000000],
#    [6., 6., 270.],
#    [6., -6., 180.],
#    [-6., -6., 90.],
#    [-6., 4., 0.000000],
#]


class Node(object):
    nodes = 1
    def __init__(self,p):
        self.position = p
        self.nodeid = Node.nodes
        self.aresta_from = []
        self.aresta_to = []
        Node.nodes += 1


    def tolist(self):
        return self.position.tolist()



xo,yo, angles = zip(*movimentos)


movimentos = [np.asarray(m) for m in movimentos]
transitions = [ movimentos[i+1]-movimentos[i] for i in range(len(movimentos)-1)]
conf = [0. for i in range(len(transitions))]
#conf[15] = 5
#conf[16] = 10
#conf[17] = 20
#transitions.append(movimentos[0]-movimentos[-2])

pts = []
for i in range(len(transitions)):
    n = Node(movimentos[i])
    n.conf = conf[i]
    if pts != []:
        pts[-1].aresta_to.append([n, transitions[i-1].tolist()])
        n.aresta_from.append([pts[-1], transitions[i-1].tolist()])
    pts.append(n)

pts[-1].aresta_to.append([pts[0], transitions[-1].tolist()])
pts[0].aresta_from.append([pts[-1], transitions[-1].tolist()])

ite = 0
plt.ion()
fig = plt.figure()
while True:
    delta_total = np.zeros(3)
    x,y, angles = zip(*[pt.tolist() for pt in pts])
    x = list(x)
    y = list(y)
    x.append(x[0])
    y.append(y[0])
    fig.clf()
    plt.plot(x,y,"b")
    plt.plot(xo,yo,"g")
#    for pt in pts:
#        for t,transition in pt.aresta_to:
#            x, y, z = zip(*[pt.position,pt.position + transition])
#            plt.plot(x,y,"r")

##    for i in range(len(angles)):
#        angle = angles[i]
#        x1 = x[i] + 1*cos(deg2rad(angle))
#        y1 = y[i] + 1*sin(deg2rad(angle))
#        plt.plot([x[i],x1],[y[i],y1],"r")
    plt.xlim(-10,10)
    plt.ylim(-10,10)
#    plt.show()
    fig.canvas.draw()
    fig.savefig("imgs/fig_{0:06d}.png".format(ite))
    fig.canvas.get_tk_widget().update() 
    print "iteration ",ite
    for pt in pts:
#        x,y, angles = zip(*[pto.tolist() for pto in pts])
#        x = list(x)
#        y = list(y)
#        x.append(x[0])
#        y.append(y[0])

#        fig.clf()
#        plt.plot(x,y,"b")
#        plt.plot(xo,yo,"g")
#        print pt.nodeid
#        fig.canvas.draw()
#        fig.savefig("imgs/fig_{0:06d}.png".format(ite))
#        fig.canvas.get_tk_widget().update() 
#        ite +=1
        sum_from = np.zeros(3)
        for f,transition in pt.aresta_from:
#            print "FROM: ", f.nodeid
            sum_from += pt.position - f.position - transition
            
        sum_to = np.zeros(3)
        for t,transition in pt.aresta_to:
#            print "TO: ", t.nodeid
            sum_to += t.position - pt.position - transition
#            print "FROM: ", t.position, pt.position, transition
#        print "SUM_FROM: ", sum_from
#        print "TOTAL: ", sum_from + sum_to
#        raw_input()
#        delta = 0.5*pt.conf/20*(sum_from + sum_to)
        delta = .5*(sum_to - sum_from)
        print "Position: ", pt.position
        print "DELTA: ", delta
        pt.position += delta
        angle = -deg2rad(delta[2])
        
#        for i in range(len(pt.aresta_from)):
#            f, transition = pt.aresta_from[i]
##            print transition, delta
##            raw_input()
#            transition_new = (np.matrix([
#                [cos(angle), -sin(angle), 0.],
#                [sin(angle), cos(angle), 0.],
#                [0., 0., 1.],
#            ]) * np.matrix(transition).T).T.A.reshape((3,))
#            pt.aresta_from[i][1] = transition_new
#            node_from = pt.aresta_from[i][0]
#            index = node_from.aresta_to.index([pt, transition])
#            node_from.aresta_to[index][1] = transition_new
        
        for i in range(len(pt.aresta_to)):
            t, transition = pt.aresta_to[i]
#            print transition, delta
#            raw_input()
            transition_new = (np.matrix([
                [cos(angle), -sin(angle), 0.],
                [sin(angle), cos(angle), 0.],
                [0., 0., 1.],
            ]) * np.matrix(transition).T).T.A.reshape((3,))
            pt.aresta_to[i][1] = transition_new
            node_to = pt.aresta_to[i][0]
            index = node_to.aresta_from.index([pt, transition])
            node_to.aresta_from[index][1] = transition_new
        delta_total += delta
    delta_total = abs(delta_total)
#    move = [-6., 6., 0.000000]-pts[0].position
#    move[2] = 0.0
#    for pt in pts:
#        pt.position += move
    ite += 1
#    if delta_total[0] < 1e-3 and delta_total[1] < 1e-3 or ite == 1000000000: #
#        break



