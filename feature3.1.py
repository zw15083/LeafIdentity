import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)
from EM import CutOut
from scipy import ndimage

def ExtractReg(angles,radius,n):
    fit = np.polyfit(angles,radius,n)
    fitfn = np.poly1d(fit)
    ang = np.linspace(0,360,360)
    rad = fitfn(ang) 
    error = abs(radius - fitfn(angles))
    return ang,rad,fit,fitfn,error



def main():
    
    img=cv2.imread('zoe/wb1127-05-2.jpg')
    x1,s1,x2,s2 = CutOut(img)
    thresh = 255*s2.astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

    nnz=np.nonzero(edges)
    z=len(nnz[0])
    plt.imshow(thresh)
    plt.show()    
    plt.imshow(edges)
    plt.show()
    #FIND CENTROID
    '''
    important: centroid is done on original image (before edge detection is  
    made), otherwise centroid is shifted.
    '''
    sourceIm,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,\
                                                    cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])       
    
    ##normalised vector
    ax=cx-20   #a point with same row but different col
    ay=cy
    ux=ax-cx
    uy=ay-cy
    u=[ux,uy]
    
    #edge vector
    vx = nnz[1] - cx
    vy = nnz[0] - cy
    
    #find angle and radius of each edge point
    angles=np.zeros(z)
    radius=np.zeros(z)
    for i in range(0,z):
        #find angle
        if nnz[0][i]>cy:#if edge point is under norm vector, add 180 deg
             angles[i]=np.arccos(np.dot(u,[vx[i],vy[i]])/(norm(u)*norm([vx[i] \
                   ,vy[i]])))*(180/np.pi)+180
             
        else:
             angles[i]=np.arccos(np.dot(u,[vx[i],vy[i]])/(norm(u)*norm([vx[i] \
                   ,vy[i]])))*(180/np.pi)
        
        #find radius
        radius[i]=norm([vx[i],vy[i]])
        
        
        #plot radius vs angle
        
    plt.scatter(angles, radius,0.1,'b')
    
    ##################################
    n=20
    ang,rad,fit,fitfn,error = ExtractReg(angles,radius,n)
    print(fit)
    print(np.mean(error))
    plt.plot(ang, rad, 'r')
    plt.scatter(angles,error,0.1,'k')
    ###################################
    
    
    plt.show()
    
    return fit,np.mean(error)
    
    
if __name__ == "__main__":
    main()