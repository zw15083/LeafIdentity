import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)

def main():
    img=cv2.imread('leaf1.jpg')
    
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    edges=cv2.Canny(thresh,100,200)
    
    nnz=np.nonzero(edges)
    z=len(nnz[0])
    
    #FIND CENTROID
    '''
    important: centroid is done on original image (before edge detection is made), otherwise centroid is shifted.
    '''
    sourceIm,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])       
    
    img[cy,:]=255
    img[:,cx]=255
          
        
    cv2.imshow('original1',img)
    ##normalised vector
    utop=[-20,0]
    ubot=[20,0]
    
    #edge vector
    vx = nnz[1] - cx
    vy = nnz[0] - cy
    
    #find angle and radius of each edge point
    angles=np.zeros(z)
    radius=np.zeros(z)
    for i in range(0,z):
        #find angle
        if nnz[0][i]>cy:#if edge point is under norm vector, add 180 deg
             angles[i]=np.arccos(np.dot(ubot,[vx[i],vy[i]])/(norm(ubot)*norm([vx[i],vy[i]])))*(180/np.pi)+180
             
        else:
             angles[i]=np.arccos(np.dot(utop,[vx[i],vy[i]])/(norm(utop)*norm([vx[i],vy[i]])))*(180/np.pi)
        
        #find radius
        radius[i]=norm([vx[i],vy[i]])
    #print(cx,cy)    
    print(np.argmax(radius))
#    arg=np.argmax(radius)
#    angNorm=angles[arg]
##    print(angles)
#    print(angles) 
#    angles[arg:]-=angNorm
#    angles[:arg]+=360-angNorm
#    #print(angles)
#        #plot radius vs angle
#    #plt.plot(angles,radius,'.')  
#    #print(max(radius))
#    #angles=np.roll(angles,len(radius)-arg)
#    radius=np.roll(radius,len(radius)-arg)
#    print(radius[0])
    #print(angles)
    #plt.figure()  
    plt.plot(angles,radius,'.')  
    
    
    
    
if __name__ == "__main__":
    main()