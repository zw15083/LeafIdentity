from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *
import os
import cv2
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from skimage.morphology import erosion, dilation
from skimage.morphology import disk
from scipy import ndimage

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


    if gmm.means_[1][1] > gmm.means_[0][1]:
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
#    plt.xlabel('Saturation')
#    plt.ylabel('Value')
#    plt.contour(X, Y, Z)
#    plt.show()
#    
#    print(gmm.score_samples)
    
    p1 = multivariate_normal(mu1, cov1)
    p2 = multivariate_normal(mu2, cov2)
    
    p1 = p1.pdf(P)
    p2 = p2.pdf(P)
    
    x1 = np.heaviside(p1-p2,0)
    
    x2 = ndimage.binary_fill_holes(x1).astype(int)
    
    s1 = erosion(x1, disk(3))
    s1 = dilation(s1, disk(3))
    
    s2 = erosion(x2, disk(3))
    s2 = dilation(s2, disk(3))

    return x1,s1,x2,s2
#    
##y = imread('wb1127-03-2.jpg')
#input_path = './Leaf_Samples/'
##output_pathx1 = './output_segmentsx1/'
##output_paths1 = './output_segmentss1/' 
##output_pathx2 = './output_segmentsx2/'
#output_paths2 = './output_segmentss2/' 
##
#folders = next(os.walk(input_path))[1]
##
#for folder in folders:
#    files = next(os.walk(input_path+folder))[2]
#	
#	#create the output folder if it doesn't exist
#    if not os.path.exists(output_paths2 + folder):
#        os.makedirs(output_paths2 + folder)
#	
#	#go through all the images under this label (folder)
#    for file in files:
#        print('yay')
#        y = imread(input_path+folder+'/'+file)
#        x1,s1,x2,s2 = CutOut(y)
##        imsave(output_pathx1 + folder + '/' + file, x1)
##        imsave(output_paths1 + folder + '/' + file, s1)
##        imsave(output_pathx2 + folder + '/' + file, x2)
#        imsave(output_paths2 + folder + '/' + file, s2)

		
