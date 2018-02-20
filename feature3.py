import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
from feature3_1 import ExtractReg
np.set_printoptions(threshold=np.inf)

def contour(im_norm):
    # Contour of leaf on plain background, thickness 10
    ret, thresh = cv2.threshold(im_norm, 127, 255, 0)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea) # draws largest contour

    leaf_contour = cv2.drawContours(np.zeros(im_norm.shape), [c], 0, (255, 255, 255),8)
    #note, argument = -1 draws all contours
    #con, leaf_contour = contour(im_norm)
    con=np.reshape(c,[c.shape[0],2])
    print(con)
    #np.savetxt('contourvalues',c)
    return con, leaf_contour

def main():
    img=cv2.imread('dummyPics/weedcolour.jpg')
    
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    a,b=contour(grey)
    #print(np.shape(a))
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
    
    
    #sort radius based on increasing angles
    myZip=list(zip(angles,radius))
    RS= [ x[1] for x in sorted(myZip) ]  #RS means radiusSorted
    
    #start radius list with max value     
    radiusSorted=np.concatenate([RS[np.argmax(RS):],RS[:np.argmax(RS)]])
    
    plt.figure()  
    
    #plt.plot(sorted(angles),radiusSorted,'.')  
    plt.plot(angles,radius,'.') 
   
    ##########
    
    
    
    n=17
    ang,rad,fit,fitfn,error = ExtractReg(angles,radius,n)
    #print(fit)
    #print(np.mean(error))
    plt.plot(ang, rad, 'r')
    #print(rad)
    #plt.scatter(angles,error,0.1,'k')
     
    
    
    
if __name__ == "__main__":
    main()