import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)

def main():
    img=cv2.imread('weedcolour3.jpg')
    
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    edges=cv2.Canny(thresh,100,200)
    
    nnz=np.nonzero(edges)
    
    #plt.plot(nnz[1],nnz[0],'.')#visualise the boundaries
    
    #####FIND CENTROID
    sourceIm,contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(np.shape(contour[0]))
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    M = cv2.moments(cnt)
    cCol = int(M['m10']/M['m00'])
    cRow = int(M['m01']/M['m00'])       
    ###### 
    
    deltaY = nnz[0] - cRow;
    deltaX = nnz[1] - cCol;
    #print(cx,cy)
    #print(deltaX)
    #print(np.where(nnz[1] == cCol))
    #print(cCol)
    angles=np.zeros(len(deltaY))
    for i in range(0,len(deltaY)):
        if deltaX[i]!=0:
            angles[i] = np.arctan(deltaY[i]/deltaX[i])
        elif nnz[0][i]>cRow:
            angles[i]=90
        else:
            angles[i]=-90
        
    print(sorted(angles*100))       
    
      
    '''visualise centroid set to 255 for white lines
    edges[127,:]=0 
    edges[:,128]=0  
    '''    
    #cv2.imshow('edges', edges)
    
    
if __name__ == "__main__":
    main()