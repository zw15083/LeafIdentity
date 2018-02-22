import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)
from EM import CutOut
from scipy import ndimage

def contour(thresh):
    # Contour of leaf on plain background, thickness 10


    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea) # draws largest contour

    leaf_contour = cv2.drawContours(np.zeros(thresh.shape), [c], 0, (255, 255, 255), 8)
    #note, argument = -1 draws all contours

    return c, leaf_contour


def UseLocoEfa(s):
    
    thresh = 255*s.astype(np.uint8)
    
    c,leafcontour = contour(thresh)
    
    edges = np.zeros(thresh.shape)
    
    for i in range(c.shape[0]):
        edges[c[i][0][1]][c[i][0][0]] = 1

    nnz=np.nonzero(edges)
    
    for i in range(c.shape[0]):
        nnz[0][i] = c[i][0][0]
        nnz[1][i] = c[i][0][1]  
        
    #########################################
    # C stuff here
    ########################################
#    
    