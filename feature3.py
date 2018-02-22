import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
from scipy.signal import argrelextrema as argre
np.set_printoptions(threshold=np.inf)





def preproc(img):
    #grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grey = img
    #_,thresh = cv2.threshold(grey, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh=grey    
    edges=cv2.Canny(thresh,100,200)
    nnz=np.nonzero(edges)
    
    return thresh,edges,nnz

def findMax(x,argOrder):   
    #start radius list with min value     
    max1=np.concatenate([x[np.argmin(x):],x[:np.argmin(x)]])
    
    shift=np.argmax(max1)
    
    #find index of local max 
    localMax=argre(max1, np.greater,order=argOrder)

    pastMax=localMax-shift
    localMax=pastMax[0]

    #start radius list with max value 
    radiusArray=np.concatenate([max1[np.argmax(max1):],max1[:np.argmax(max1)]])
    localMin=argre(radiusArray, np.less,order=argOrder)   

    nOfMax=np.shape(pastMax)[1]
    localMax,localMin=radiusArray[localMax],radiusArray[localMin]
    return radiusArray,nOfMax,localMax,localMin
  
def graphNorm(x):
    maxRatio=max(x)
    y=x/maxRatio
    return(y)    
      
def maxMinDiff(bigMax,bigMin):
    #initialisation
    diffLeft=np.zeros(len(bigMax))    
    diffRight=np.zeros(len(bigMax))
    
    #first element only    
    diffLeft[0]=bigMax[0]-bigMin[-1]
    diffRight[0]=bigMax[0]-bigMin[0]

    if len(bigMax)>1:#if there are more than 1 max
      for i in range(1,len(bigMax)):
        diffLeft[i]=round(bigMax[i],2)-round(bigMin[i-1],2)
        diffRight[i]=round(bigMax[i],2)-round(bigMin[i],2)
      
    allDiff=np.concatenate([diffLeft,diffRight])    
    print(np.mean(allDiff))
    print(max(allDiff))
    

  
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
    #plt.figure()      
    #plt.plot(angles,normRS,'.') 
    
    #find local max
    newRS,nOfMax,bigMax,bigMin=findMax(normRS,40)
    # print(bigMax)
    # plt.plot(np.linspace(0,len(newRS),len(newRS)),newRS) 
    # plt.show()
    #smallMax=findMax(RS,10)
    #maxMinDiff(bigMax,bigMin)
    
    #if abs(bigMax-smallMax)>
#    print('bm=',bigMax)
#    print('sm=',smallMax)
    return nOfMax
        




    
#if __name__ == "__main__":
#    main(a,b,c)

#main(cv2.imread('LabSeg2/TrainSeg2/Acer/wb1128-01-1.jpg',0))
#main(cv2.imread('weedcolour.jpg'))