import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
from feature3_1 import ExtractReg
from EMT import CutOut
np.set_printoptions(threshold=np.inf)

def contour(thresh):
    # Contour of leaf on plain background, thickness 10


    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea) # draws largest contour

    leaf_contour = cv2.drawContours(np.zeros(thresh.shape), [c], 0, (255, 255, 255), 8)
    #note, argument = -1 draws all contours

    return c, leaf_contour



def preprocTristan():
    img=cv2.imread('dummyPics/leaf1.jpg')
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #a,b=contour(grey)
    #print(np.shape(a))
    _, thresh = cv2.threshold(grey, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    c,leafcontour = contour(thresh) 
    
    col=np.zeros(len(c))
    row=np.zeros(len(c))
    #loool=np.zeros(len(c))
    for i in range(len(c)):
      #print(c[i][0])
      col[i],row[i]=c[i][0]
      #print(c[i][0])
    #print(loool)
    ok=[row,col]
    edges=cv2.Canny(thresh,100,200)
    nnz=np.nonzero(edges)
    z=len(nnz[0])
    
    return thresh,edges,nnz,z,ok
  
def main(thresh,img,nnz,z,ok):

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
          
    cv2.imshow('thr',thresh)    
    cv2.imshow('original1',img)
    ##normalised vector
    utop=[-20,0]
    ubot=[20,0]
    
    #edge vector
    
    vx = ok[1] - cx
    vy = ok[0] - cy
    
    #find angle and radius of each edge point
    angles=np.zeros(len(nnz[0]))
    radius=np.zeros(len(nnz[0]))
    print(np.shape(vx))
    print(np.shape(nnz))
    for i in range(0,len(vx)):
        #find angle
        if nnz[0][i]>cy:#if edge point is under norm vector, add 180 deg
             angles[i]=np.arccos(np.dot(ubot,[vx[i],vy[i]])/(norm(ubot)*norm([vx[i],vy[i]])))*(180/np.pi)+180
             
        else:
             angles[i]=np.arccos(np.dot(utop,[vx[i],vy[i]])/(norm(utop)*norm([vx[i],vy[i]])))*(180/np.pi)
        
        #find radius
        radius[i]=round(norm([vx[i],vy[i]]),2)
    
    print(radius)
    #sort radius based on increasing angles
    myZip=list(zip(angles,radius))
    RS= [ x[1] for x in sorted(myZip) ]  #RS means radiusSorted
    
    #start radius list with max value     
    radiusSorted=np.concatenate([RS[np.argmax(RS):],RS[:np.argmax(RS)]])
    
    plt.figure()  
    #print(len)
    #plt.plot(sorted(angles),radiusSorted,'.')  
    plt.plot(angles,radius,'.') 
   
    ##########
  
#    n=17
#    ang,rad,fit,fitfn,error = ExtractReg(angles,radius,n)
#    plt.scatter(angles, radius,0.1,'b',linewidths=2)
#    #print(radius)
#    plt.figure() 
#    plt.plot(angles,radius,'.') 
    return radius
    


        
a,b,c,d,ok=preprocTristan()



    
#if __name__ == "__main__":
#    main(a,b,c,d)

main(a,b,c,d,ok)