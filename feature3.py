import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
from feature3_1 import ExtractReg
from EMT import CutOut
from scipy.signal import argrelextrema as argre
np.set_printoptions(threshold=np.inf)





def preprocTristan():
    img=cv2.imread('dummyPics/leaf1.jpg')
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)    
    edges=cv2.Canny(thresh,100,200)
    nnz=np.nonzero(edges)
    z=len(nnz[0])
    
    return thresh,edges,nnz,z
  
def main(thresh,img,nnz,z):

    #FIND CENTROID
    '''
    important: centroid is done on original image (before edge detection is made), otherwise centroid is shifted.
    '''
    
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
   
    col=np.zeros(len(cnt))
    row=np.zeros(len(cnt))
  
    for i in range(len(cnt)):
      
      col[i],row[i]=cnt[i][0]
    ok=[row,col]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])       
    
    img[cy,:]=255
    img[:,cx]=255
          
    cv2.imshow('thr',thresh)    
    cv2.imshow('original1',img)
    
    ##normalised vector
    utop=[-20,0]
    ubot=[20,0]
    
    #edge vector   
    vx = ok[1] - cx
    vy = ok[0] - cy
    
    #find angle and radius of each edge point
    angles=np.zeros(len(vx))
    RS=np.zeros(len(vx))
    
    for i in range(0,len(vx)):
        #find angle
        if ok[0][i]>cy:#if edge point is under norm vector, add 180 deg
             angles[i]=np.arccos(np.dot(ubot,[vx[i],vy[i]])/(norm(ubot)*norm([vx[i],vy[i]])))*(180/np.pi)+180
             
        else:
             angles[i]=np.arccos(np.dot(utop,[vx[i],vy[i]])/(norm(utop)*norm([vx[i],vy[i]])))*(180/np.pi)
        
        #find radius
        RS[i]=round(norm([vx[i],vy[i]]),2)
    
    
    
    
    #start radius list with max value     
    radiusSorted=np.concatenate([RS[np.argmin(RS):],RS[:np.argmin(RS)]])
    argOrder=15
    argAll=argre(radiusSorted, np.greater,order=argOrder)
    for i in argAll:
      print(radiusSorted[i])
    print('len=',np.shape(argre(radiusSorted, np.greater,order=argOrder))[1])
    #print(RS)
    plt.figure()  
    #print(len)
    #plt.plot(sorted(angles),radiusSorted,'.')  
    plt.plot(angles,RS,'.') 
   
    ##########
  
#    n=17
#    ang,rad,fit,fitfn,error = ExtractReg(angles,radius,n)
#    plt.scatter(angles, radius,0.1,'b',linewidths=2)
#    #print(radius)
#    plt.figure() 
#    plt.plot(angles,radius,'.') 
    return RS
    


        
a,b,c,d=preprocTristan()



    
#if __name__ == "__main__":
#    main(a,b,c,d)

main(a,b,c,d)