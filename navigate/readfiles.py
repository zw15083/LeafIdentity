from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *
import glob
import os


def sep(x,y,fore,back):
	M = shape(y)[0]
	N = shape(y)[1]
	for j in range(M):
		for i in range(N):
			pix = y[j,i]
			F = fore[pix]
			B = back[pix]
			if F > B:
				x[j,i] = 1
			else:
				x[j,i] = 0
	return x

def histograms(filename,leaf,background1,background2):
	norm = 32640	
	y = imread(filename)
	Y = y[:,:,2]
	x = np.random.randint(2, size=(shape(y)[0], shape(y)[1]))
	plt.figure(1)
	histy = plt.hist(leaf.ravel(),256,[0,256])[0]
	fore = histy/norm
	hists = plt.hist(background1.ravel(),256,[0,256])[0] + plt.hist(background2.ravel(),256,[0,256])[0]
	back = hists/norm


norm = 32640
leaf = imread('leaf.jpg')
background = imread('back.jpg')
histy = plt.hist(leaf.ravel(),256,[0,256])[0]
fore = histy/norm
hists = plt.hist(background.ravel(),256,[0,256])[0] 
back = hists/norm

input_path = './leaves/'
output_path = './output_segments/'

folders = next(os.walk(input_path))[1]

for folder in folders:
	files = next(os.walk(input_path+folder))[2]
	
	#create the output folder if it doesn't exist
	if not os.path.exists(output_path + folder):
		os.makedirs(output_path + folder)
	
	#go through all the images under this label (folder)
	for file in files:
		y = imread(input_path+folder+'/'+file)
		Y = y[:,:,1]
		x = np.random.randint(2, size=(shape(y)[0], shape(y)[1]))
		X = sep(x,Y,fore,back)
		imsave(output_path + folder + '/' + file, X)
