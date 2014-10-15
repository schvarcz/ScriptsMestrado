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
             "motox/VID_20140617_162058756_GRAY_equalized_small_ESCOLHA2",
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


for matcher in matchers:
    for skip in [1]:
        for path in paths:
            for step in steps:
                ##################
                # Load from file #
                ##################
                trans, (mx,nx), (my,ny), (mz,nz) = loadFromFile(pathResultados+"/features/{0}/step{3}_{1}/posicoes_{2}.csv".format(path,step,step,matcher))

                pathSalvar = pathResultados+"/figures/{0}/step{3}_{1}/skip_{2}".format(path,step,skip,matcher)
                if(not os.path.exists(pathSalvar)):
                    os.makedirs(pathSalvar)

                graph = Graph()
                plotLimits = (mx,nx), (my,ny), (mz,nz)

                oldNode = None
                for j in xrange(0,len(trans),skip):

                    n = Node(trans[j][:-1], trans[j][-1])

                    ssdResult = float("inf")

                    if j > 385:
                        ssdResult, revisitedNode = graph.revisited(n)

                        if ssdResult < 40000000:
                            n = revisitedNode

                    if oldNode != None:
                        transition = np.asarray(trans[j][:-1]) - np.asarray(trans[j-1][:-1])
                        oldNode.aresta_to.append([n,transition])
                        n.aresta_from.append([oldNode,transition])

                    if ssdResult > 40000000:
                        graph.append(n)
                        print "novo"

                    print "Node: ", n.nodeid

                    oldNode = n

                    graph.relaxMilford()
                    robotPath = zip(*graph.tolist())
                    img = cv2.imread(pathResultados+"/features/{0}/step{3}_{1}/I1_{2:06d}.png".format(path,step,j,matcher))

                    plotVisualSLAM(robotPath, img, j, len(trans), plotLimits, "{0}/fig_{1:06d}.png".format(pathSalvar,j))

                    print j, " / ", len(trans)

                ite = 0
                delta_total = np.ones(3)
                while(ite < 1000 and (delta_total[0] > 1e-3 or delta_total[1] > 1e-3 or delta_total[3] > 1e-3)):
                    ite += 1
                    delta_total = graph.relaxMilford()
                    robotPath = zip(*graph.tolist())
                    img = cv2.imread(pathResultados+"/features/{0}/step{3}_{1}/I1_{2:06d}.png".format(path,step,j,matcher))

                    plotVisualSLAM(robotPath, img, j, len(trans), plotLimits, "{0}/fig_{1:06d}.png".format(pathSalvar,j))




