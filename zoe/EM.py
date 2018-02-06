from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *
import os
import cv2
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy.signal import convolve2d



def CutOut(y1):
    y = cv2.blur(y1,(1,1))
    hsv = cv2.cvtColor(y, cv2.COLOR_BGR2HSV)
    hsv_shape = hsv.shape
    gmm = GaussianMixture(n_components=2,covariance_type='diag')
    hsv1 = cv2.cvtColor(y1, cv2.COLOR_BGR2HSV)
    SV = np.zeros((hsv_shape[0]*hsv_shape[1],2))
    
    #lu[:,0] = hsv[:,:,0].flatten()
    Saturation = hsv[:,:,1].flatten()
    Value = hsv[:,:,2].flatten()
    
    for i in range(Saturation.size):
        if Saturation[i] != 0 and Value[i] !=0:
            SV[i,0] = Saturation[i]
            SV[i,1] = Value[i]
    
    max1 = 1#np.max(SV[:,0])
    max2 = 1#np.max(SV[:,1])
    #max3 = np.max(lu[:,2])
    SV[:,0] = SV[:,0]/max1
    SV[:,1] = SV[:,1]/max2
    #lu[:,2] = lu[:,2]/max3
    gmm.fit(SV)
    
    X, Y = np.meshgrid(np.linspace(0, 1,256), np.linspace(0,1,200))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = gmm.score_samples(XX)
    Z = Z.reshape((200,256))
    plt.scatter(SV[:, 0], SV[:, 1],0.25,'b')
    plt.contour(X, Y, Z)
    plt.show()
    
    x = np.random.randint(2, size=(shape(y)[0], shape(y)[1]))
    
    M = shape(y)[0]
    N = shape(y)[1]
    
    MeanS = gmm.means_[:][0]
    
    if MeanS[0] > MeanS[1]:
        mu1 = gmm.means_[0]
        mu2 = gmm.means_[1]
        cov1 = gmm.covariances_[0]
        cov2 = gmm.covariances_[1]
    else:
        mu2 = gmm.means_[0]
        mu1 = gmm.means_[1]        
        cov2 = gmm.covariances_[0]
        cov1 = gmm.covariances_[1]    
    
    
    
    for j in range(M):
        for i in range(N):    
            p1 = multivariate_normal.pdf(np.array([hsv1[j,i,1]/max1,hsv1[j,i,2]/max2]), mu1, \
                     cov1, allow_singular=False)
            p2 = multivariate_normal.pdf(np.array([hsv1[j,i,1]/max1,hsv1[j,i,2]/max2]), mu2, \
                     100*cov2, allow_singular=False) 
            if p1 > p2:
                x[j,i] = 1
            else:
                x[j,i] = 0

    x = erosion(x, disk(5))
    x = dilation(x, disk(4))
    plt.imshow(x)
    plt.show()

    s = skeletonize(x == 1)


    d = y1
    d[:,:,0] = x*y1[:,:,0]
    d[:,:,1] = x*y1[:,:,1]
    d[:,:,2] = x*y1[:,:,2]

    return x,d,s
    

y1 = imread('ny1053-04-2.jpg')
plt.imshow(y1)
plt.show()
x,d,s = CutOut(y1)

plt.imshow(s)
plt.show()

plt.imshow(d)
plt.show()

#y = imread('wb1127-03-2.jpg')
#input_path = './Leaf_Samples/'
#output_path = './output_segments/'
#
#folders = next(os.walk(input_path))[1]
#
#for folder in folders:
#    files = next(os.walk(input_path+folder))[2]
#	
#	#create the output folder if it doesn't exist
#    if not os.path.exists(output_path + folder):
#        os.makedirs(output_path + folder)
#	
#	#go through all the images under this label (folder)
#    for file in files:
#        print('yay')
#        y = imread(input_path+folder+'/'+file)
#        y = y[1:550,1:550,:]
#        x,d = CutOut(y)
#        imsave(output_path + folder + '/' + file, x)

		
