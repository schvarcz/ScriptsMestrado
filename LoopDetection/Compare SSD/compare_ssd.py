import numpy as np
from os import listdir, path
from cv2 import *

pathImgs = path.expanduser("~/Dissertacao/datasets/2010_03_09_drive_0019/")
img2Compare = "I1_000127.png"


def ssd(template,comp):
    sd = np.power(comp-template,2)
    return sd.sum()
    
def sad(template,comp):
    sd = comp-template
    return sd.sum()


toFind = imread(pathImgs+img2Compare)[100:250,400:1000]
minimo = float("inf"), ""

for imgName in sorted(listdir(path.expanduser(pathImgs))):
    if img2Compare == imgName or not imgName.endswith("png"):
        continue
    toCompare = imread(pathImgs+imgName)[100:250,400:1000]
    result = ssd(toFind, toCompare)
    if result < minimo[0]:
        minimo = result, imgName

print minimo
