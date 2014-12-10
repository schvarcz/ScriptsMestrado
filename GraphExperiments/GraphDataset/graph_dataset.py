# -*- coding: utf8 -*-
import os, cv2
from visualSLAM import *
from utils import *
from plotter import *

paths = [ 
#            "2010_03_09_drive_0019",
#            "drone/20140318_132620",
#            "drone/20140318_132620_gray",
#            "drone/20140318_133931",
#            "drone/20140318_133931_gray",
#            "drone/20140327_135316_gray",
#            "drone/20140328_102444_gray",
#             "motox/VID_20140617_163505406_GRAY",
#             "motox/VID_20140617_162058756_GRAY",
#             "motox/VID_20140617_162058756_GRAY_ESCOLHA",
#             "motox/VID_20140617_162058756_GRAY_equalized",
#             "motox/VID_20140617_162058756_GRAY_CURVA",
#             "motox/VID_20140617_162058756_GRAY_small_CURVA",
#             "motox/VID_20140617_162058756_GRAY_equalized_small_CURVA",
#             "motox/VID_20140617_162058756_GRAY_equalized_small",
#             "motox/VID_20140617_162058756_GRAY_equalized_ESCOLHA",
#             "motox/VID_20140617_162058756_GRAY_equalized_small_ESCOLHA2",
             "motox/VID_20140617_162058756_ESCOLHA",
#             "motox/VID_20140617_162058756_GRAY_equalized_small_GRAMA_ESCOLHA",
#             "motox/VID_20140617_162058756_GRAY_ESCOLHA_GRAMA",
#            "nao/nao2",
#            "nao/nao2_gray",
#            "nao/nao2_rect",
#            "nao/nao2_rect_escolha",
#            "nao/naooo_2014-03-10-17-48-35",
#            "nao/naooo_2014-03-10-17-48-35_gray"
            ]

matchers = [
#        "",
#        "_eq_match_sift_constrainted",
        "_eq_match_sift",
#        "_match_sift",
#        "_match_surf",
#        "_match_surf_sift",
#        "_knn_sift",
#        "_knn_surf",
#        "_knn_surf_sift",
#        "_knn_sift_closer",
#        "_knn_surf_closer",
#        "_knn_surf_sift_closer"
        ]

steps = [1]#,2,3,5,10,15,20]

pathResultados = os.path.expanduser("~/Dissertacao/OdometriaVisual/")

ssd_conf = 90000000
for matcher in matchers:
    for skip in [1]:
        for path in paths:
            for step in steps:
                ##################
                # Load from file #
                ##################
                trans, (mx,nx), (my,ny), (mz,nz) = loadFromFile(pathResultados+"/features/{0}/step{3}_{1}/posicoes_{2}.csv".format(path,step,step,matcher))

                pathSalvar = pathResultados+"/graphs/{0}/step{3}_{1}_ssd/skip_{2}".format(path,step,skip,matcher)
                if(not os.path.exists(pathSalvar)):
                    os.makedirs(pathSalvar)

                graph = Graph()
                plotLimits = (mx,nx), (my,ny), (mz,nz)

                oldNode = None
                for j in xrange(0,421,skip):

                    n = Node(trans[j][:-1], trans[j][-1])

                    ssdResult = float("inf")

                    if j > 385:
                        ssdResult, revisitedNode = graph.revisited(n)
                        #ssdResult, revisitedNode = 50, graph[j-385]

                        if ssdResult < ssd_conf:
                            n = revisitedNode

                    if oldNode != None:
                        transition = (np.asarray(trans[j][:-1]) - np.asarray(trans[j-1][:-1])).tolist()
                        oldNode.aresta_to.append([n,transition])
                        n.aresta_from.append([oldNode,transition])

                    if ssdResult > ssd_conf:
                        graph.append(n)
                        print "novo"

                    if n.pose[4] > 0.04:
                        n.conf = 4
                    print "Node: ", n.nodeid

                    oldNode = n

                    if j>385:
                        graph.relaxMine()
                        robotPath = zip(*graph.tolist())
                        img = cv2.imread(pathResultados+"/features/{0}/step{3}_{1}/I1_{2:06d}.png".format(path,step,j,matcher))

                        plotVisualSLAM(robotPath, img, j, len(trans), plotLimits)#, "{0}/fig_{1:06d}.png".format(pathSalvar,j))

                    print j, " / ", len(trans)

                ite = 0
                delta_total = np.ones(3)
                while True: #(ite < 1000 and (delta_total[0] > 1e-3 or delta_total[1] > 1e-3 or delta_total[3] > 1e-3)):
                    ite += 1
                    print "Iteration: ", ite
                    delta_total = graph.relaxMine()
                    robotPath = zip(*graph.tolist())
                    img = cv2.imread(pathResultados+"/features/{0}/step{3}_{1}/I1_{2:06d}.png".format(path,step,j-1,matcher))

                    plotVisualSLAM(robotPath, img, j+ite, len(trans), plotLimits, "{0}/fig_{1:06d}.png".format(pathSalvar,j+ite))




