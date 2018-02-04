from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from PIL import Image

norm = 32640

y = imread('wb1127-05-2.jpg')

Y = y[:,:,1]
x = np.random.randint(2, size=(shape(y)[0], shape(y)[1]))
leaf = imread('leaf.jpg')
#background2 = imread('back2.jpg')
background3 = imread('back3.jpg')
#looking at making images black and white
plt.figure(1)
col = Image.open("wb1127-03-2.jpg")
gray = col.convert('L')
bw = gray.point(lambda x: 0 if x<128 else 255, '1')
bw.save("leafy.jpg")
plt.imshow(bw)
plt.show()
plt.figure(2)
histy = plt.hist(leaf.ravel(),256,[0,256])[0]
fore = histy/norm
hists = plt.hist(background3.ravel(),256,[0,256])[0] # + plt.hist(background2.ravel(),256,[0,256])[0]
back = hists/norm
plt.show()

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


x = sep(x,Y,fore,back)	

			
plt.figure(3)
plt.imshow(x)

plt.show()