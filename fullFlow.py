import sys,os
from matplotlib import pyplot as plt
import numpy as np
import cv2

from Utils.utils import getSpotContour,delete3header,produce_patches,displayOutput
from Utils.config import configure


conf = configure()

if len (sys.argv) > 1 :
    conf.dataName = sys.argv[1]

if len (sys.argv) > 2 :
    conf.priority = int(sys.argv[2])
print(conf.priority)

outFolder = "./" + os.path.splitext(os.path.basename(conf.dataName))[0]
if not os.path.exists(outFolder):
    os.makedirs(outFolder)



img,patch,contours = getSpotContour(conf, outFolder)

yheader,top = delete3header(patch,contours,conf,outFolder)




patchesInfoResult = produce_patches(conf,img,contours,yheader,top,outFolder)

displayOutput(patchesInfoResult,outFolder)
    
