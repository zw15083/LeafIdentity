import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)

#cv2.imwrite('myTest.jpg',edges)

def main(img):
  
    shape=np.shape(img)

    topRow=0
    botRow=int(shape[0]-1)
    leftCol=0
    rightCol=int(shape[1]-1)
    
    lu=botRow
    ld=topRow
    lr=leftCol
    ll=rightCol
    
    while sum(img[lu,:]) <510: #>510 because there will a few white points due to non perfect image, 
        lu-=1                  #hence stop when more than 2 white pixels (2*255=510)
    while sum(img[ld,:]) <510:
        ld+=1  
    while sum(img[:,ll]) <510:
        ll-=1
    while sum(img[:,lr]) <510:
        lr+=1
    
  
    
    #img[lu,lr:ll]=255 
    #img[ld,lr:ll]=255
    #img[ld:lu,ll]=255
    #img[ld:lu,lr]=255   
    #cv2.imshow('a',img)
    total_area=(lr-ll)*(ld-lu)
    total_leaf=0
    for row in range(ld,lu):
        total_leaf+=sum(img[row,:])
    
    ratio= total_leaf/255/total_area

    return(ratio)
    
    
    
    
    
if __name__ == "__main__":
    main()