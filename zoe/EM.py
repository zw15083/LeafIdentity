from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *
import os
import cv2
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

norm = 32640
#y = imread('wb1127-03-2.jpg')
input_path = './Leaf_Samples/'
output_path = './output_segments/'


def CutOut(y):
	hsv = cv2.cvtColor(y, cv2.COLOR_BGR2HSV)
	hsv_shape = hsv.shape
	gmm = GaussianMixture(n_components=2,covariance_type='diag')
    
	SV = np.zeros((hsv_shape[0]*hsv_shape[1],2))
    
	#lu[:,0] = hsv[:,:,0].flatten()
	Saturation = hsv[:,:,1].flatten()
	Value = hsv[:,:,2].flatten()
	
    
	for i in range(Saturation.size):
		if Saturation[i] != 0 and Value[i] !=0:
			SV[i,0] = Saturation[i]
			SV[i,1] = Value[i]
    
	max1 = np.max(SV[:,0])
	max2 = np.max(SV[:,1])
	#max3 = np.max(lu[:,2])
	SV[:,0] = SV[:,0]/max1
	SV[:,1] = SV[:,1]/max2
	#lu[:,2] = lu[:,2]/max3
	gmm.fit(SV)
    
	X, Y = np.meshgrid(np.linspace(0, 1,256), np.linspace(0,1,200))
	XX = np.array([X.ravel(), Y.ravel()]).T
	Z = gmm.score_samples(XX)
	Z = Z.reshape((200,256))
	# plt.scatter(SV[:, 0], SV[:, 1],0.25,'b')
	# plt.contour(X, Y, Z)
	# plt.show()
    
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
			p1 = multivariate_normal.pdf(np.array([hsv[j,i,1]/max1,hsv[j,i,2]/max2]), mu1, \
					cov1, allow_singular=False)
			p2 = multivariate_normal.pdf(np.array([hsv[j,i,1]/max1,hsv[j,i,2]/max2]), mu2, \
					cov2, allow_singular=False) 
			if p1 < p2:
				x[j,i] = 1
			else:
				x[j,i] = 0
			
                
	if sum([x[1,1],x[-1,1],x[1,-1],x[-1,-1]]) >= 1:
		x = 1-x
                
                
	# plt.imshow(x)
	# plt.show()
    
	# x2 = np.zeros(y.shape)
	# x2[:,:,0] = x
	# x2[:,:,1] = x
	# x2[:,:,2] = x
	# d = x2*y
	# d = d.astype(np.uint8)
	# plt.imshow(d)
	# plt.show()
    
    
    
	return x

#y = CutOut(y)

folders = next(os.walk(input_path))[1]

for folder in folders:
	files = next(os.walk(input_path+folder))[2]
	
	#create the output folder if it doesn't exist
	if not os.path.exists(output_path + folder):
		os.makedirs(output_path + folder)
	
	#go through all the images under this label (folder)
	for file in files:
		print('yay')
		x = imread(input_path+folder+'/'+file)
		y = CutOut(x)
		imsave(output_path + folder + '/' + file, y)
