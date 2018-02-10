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



def CutOut(y):
    hsv = cv2.cvtColor(y, cv2.COLOR_BGR2HSV)
    hsv_shape = hsv.shape
    gmm = GaussianMixture(n_components=2,covariance_type='diag')
    SV = np.zeros((hsv_shape[0]*hsv_shape[1],2))
    

    Saturation = hsv[:,:,1].flatten()
    Value = hsv[:,:,2].flatten()
    P = []

    SV[:,0] = Saturation
    SV[:,1] = Value
    
    max1 = np.max(Saturation[:])
    max2 = np.max(Value[:])

    SV[:,0] = SV[:,0]/max1
    SV[:,1] = SV[:,1]/max2

    gmm.fit(SV)
    
    
    for i in range(shape(y)[0]):
        p = []
        for j in range(shape(y)[1]):
            p.append([hsv[i,j,1]/max1,hsv[i,j,2]/max2])
        P.append(p)

    MeanS = gmm.means_[:][0]
    
    if MeanS[0] > MeanS[1]:
        mu1 = gmm.means_[0]
        mu2 = gmm.means_[1]
        gmm.covariances_[1] = 100*gmm.covariances_[1]
        cov1 = gmm.covariances_[0]
        cov2 = gmm.covariances_[1]
    else:
        mu2 = gmm.means_[0]
        mu1 = gmm.means_[1] 
        gmm.covariances_[0] = 100*gmm.covariances_[0]
        cov2 = gmm.covariances_[0]
        cov1 = gmm.covariances_[1]    
    
#    X, Y = np.meshgrid(np.linspace(0, 1,256), np.linspace(0,1,200))
#    XX = np.array([X.ravel(), Y.ravel()]).T
#    Z = gmm.score_samples(XX)
#    Z = Z.reshape((200,256))
#    plt.scatter(SV[:, 0], SV[:, 1],0.25,'b')
#    plt.contour(X, Y, Z)
#    plt.show()

    p1 = multivariate_normal(mu1, cov1)
    p2 = multivariate_normal(mu2, cov2)
    
    p1 = p1.pdf(P)
    p2 = p2.pdf(P)
    
    x = np.heaviside(p1-p2,0)

    s = erosion(x, disk(5))
    s = dilation(s, disk(4))
    plt.imshow(x)
    plt.show()
    
    return x,s
    

y1 = imread('ny1053-04-2.jpg')


plt.imshow(y1)
plt.show()


x,s = CutOut(y1)

plt.imshow(s)
plt.show()


#
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

		
