from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import cv2
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

norm = 32640
y = imread('13001155900043.jpg')
x = np.random.randint(2, size=(shape(y)[0], shape(y)[1]))

hsv = cv2.cvtColor(y, cv2.COLOR_BGR2HSV)
hsv_shape = hsv.shape
gmm = GaussianMixture(n_components=2,covariance_type='full')

lu = np.zeros((hsv_shape[0]*hsv_shape[1],2))

lu[:,0] = hsv[:,:,1].flatten()
lu[:,1] = hsv[:,:,2].flatten()
max1 = np.max(lu[:,0])
max2 = np.max(lu[:,1])
lu[:,0] = lu[:,0]/max1
lu[:,1] = lu[:,1]/max2
gmm.fit(lu)

print(gmm.means_)
print('\n')
print(gmm.covariances_)

X, Y = np.meshgrid(np.linspace(0, 1,256), np.linspace(0,1,200))
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)
Z = Z.reshape((200,256))
plt.scatter(lu[:, 0], lu[:, 1],0.25,'b')
plt.contour(X, Y, Z)
plt.show()

x = np.random.randint(2, size=(shape(y)[0], shape(y)[1]))

M = shape(y)[0]
N = shape(y)[1]

for j in range(M):
    for i in range(N):    
        p1 = multivariate_normal.pdf(np.array([hsv[j,i,1]/max1,hsv[j,i,2]]/max2), gmm.means_[0], \
                 gmm.covariances_[0], allow_singular=False)
        p2 = multivariate_normal.pdf(np.array([hsv[j,i,1]/max1,hsv[j,i,2]]/max2), gmm.means_[1], \
                 gmm.covariances_[1], allow_singular=False) 
        if p1 > p2:
            x[j,i] = 1
        else:
            x[j,i] = 0
        
plt.imshow(x)
plt.show()
