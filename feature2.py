import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
import imghdr
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)

#cv2.imwrite('myTest.jpg',edges)

def main():
    img=cv2.imread('weedcolour.jpg')
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(grey, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    sourceIm,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(np.shape(contour[0]))
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(cnt)   
    drawing = np.zeros(img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)
    #all_img = np.hstack((drawing, img))
    cv2.imshow('Contours', drawing)
    print(hull)
    
if __name__ == "__main__":
    main()