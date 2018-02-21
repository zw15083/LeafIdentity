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





def preproc(img):
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(grey, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)    
    edges=cv2.Canny(thresh,100,200)
    nnz=np.nonzero(edges)
    
    return thresh,edges,nnz

def findMax(x,argOrder):
    #start radius list with min value     
    max1=np.concatenate([x[np.argmin(x):],x[:np.argmin(x)]])
    shift=np.argmax(max1)
    
    #find index of local max 
    localMax=argre(max1, np.greater,order=argOrder)
#    print(localMax)
#    print(shift)
    pastMax=localMax-shift
    print(pastMax)
    #start radius list with max value 
    min1=np.concatenate([max1[np.argmax(max1):],max1[:np.argmax(max1)]])
    localMin=argre(min1, np.less,order=argOrder)
    
    
    nOfMax=np.shape(pastMax)[1]
    return min1,nOfMax,pastMax,localMin
  
def graphNorm(x):
    maxRatio=max(x)
    y=x/maxRatio
    return(y)    
      
#def maxMinDiff(x,localMax):
    

  
def main(img):
    thresh,edges,nnz=preproc(img)
    #FIND CENTROID    
    #important: centroid is done on original image (before edge detection is made), otherwise centroid is shifted.
        
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
          
    #cv2.imshow('thr',thresh)    
    #cv2.imshow('original1',img)
    
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
        
    normRS=graphNorm(RS)
    plt.figure()      
    plt.plot(angles,normRS,'.') 
    
    #find local max
    lol,_,bigMax,bigMin=findMax(normRS,20)
    #smallMax=findMax(RS,10)
    
#    print(bigMax)
    print('max=',lol[bigMax])
    print('min=',lol[bigMin])
    #if abs(bigMax-smallMax)>
#    print('bm=',bigMax)
#    print('sm=',smallMax)
    return bigMax
        




    
#if __name__ == "__main__":
#    main(a,b,c)

main(cv2.imread('dummyPics/leaf1.jpg'))