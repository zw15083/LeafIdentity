import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)

def main():
    img=cv2.imread('weedcolour.jpg')
    
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    edges=cv2.Canny(thresh,100,200)
    #grey[:,538]=255
    #grey[298,:]=255
    
    #cv2.imshow('edges', grey)
    nnz=np.nonzero(edges)
    
    #print(len(nnz[1]))
    
    #plt.plot(nnz[1],nnz[0],'.')#visualise the boundaries
    #'''
    #####FIND CENTROID
    '''
    important: centroid is done on original image (before edge detection is made), otherwise centroid is shifted.
    '''
    sourceIm,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(np.shape(contour[0]))
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])       
    ###### 
    #edges[cy,:]=255
    #edges[:,cx]=255
    #cv2.imshow('edges', edges)
    ##arbitrary point
    ax=cx-20
    ay=cy
    
    ##normalised vector
    ux=ax-cx
    uy=ay-cy
    u=[ux,uy]
    #edge vector
    vx = nnz[1] - cx
    vy = nnz[0] - cy
    print(cx,cy)
    
    
    angles=np.zeros(len(nnz[0]))
    radius=np.zeros(len(nnz[0]))
    for i in range(0,len(nnz[0])):
        angles[i]=np.arccos(np.dot(u,[vx[i],vy[i]])/(norm(u)*norm([vx[i],vy[i]])))
        radius[i]=norm([vx[i],vy[i]])
        #if norm([vx[i],vy[i]])==0:
            #print(vx[i],vy[i])
            #print(nnz[0][i],nnz[1][i])
    #print(angles[100]*(180/np.pi))
    #'''     
        
   
      
    #distances = np.sqrt((nnz[0] - cRow)**2 + (nnz[1] - cCol)**2)
    plt.plot(angles*(180/np.pi),radius,'.')  
    #'''
    '''visualise centroid set to 255 for white lines
    edges[127,:]=0 
    edges[:,128]=0  
    #'''    
    #cv2.imshow('edges', edges)
    
    
if __name__ == "__main__":
    main()