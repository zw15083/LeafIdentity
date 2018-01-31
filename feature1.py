import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)

#cv2.imwrite('myTest.jpg',edges)

def main():
    #data='Leaf_Samples/acer_campestre/ny1079-01-2.jpg'
    data='myTestFilled.jpg'
    img=cv2.imread(data,0)
    edges=cv2.Canny(img,100,200)
    shape=np.shape(img)
    
    midRow=int(shape[0]/2)
    midCol=int(shape[1]/2)
    
    lu=midRow
    ld=midRow
    lr=midCol
    ll=midCol
    
    while sum(img[lu,:]) >510: #>510 because there will a few white points due to non perfect image, 
        lu-=1                  #hence stop when less than 2 white pixels (2*255=510)
    while sum(img[ld,:]) >510:
        ld+=1  
    while sum(img[:,ll]) >510:
        ll-=1
    while sum(img[:,lr]) >510:
        lr+=1
    
  
    
    img[lu,ll:lr]=255 
    img[ld,ll:lr]=255
    img[lu:ld,ll]=255
    img[lu:ld,lr]=255   

    total_area=(lr-ll)*(ld-lu)
    print('total_area=',total_area)
    
    total_leaf=0
    for row in range(lu,ld):
        total_leaf+=sum(img[row,:])
    print('totalLeaf=',total_leaf/255)
    ratio= total_leaf/255/total_area
    print('ratio=',ratio)
    cv2.imshow('img1',img)
    
    
    
    
    
if __name__ == "__main__":
    main()