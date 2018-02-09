import cv2
from cv2 import moments
import numpy as np

def FindCom(x):
    x = x.astype(np.uint8)
    M = cv2.moments(x)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX,cY

cX,cY = FindCom(x)