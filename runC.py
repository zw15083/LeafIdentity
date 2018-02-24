from subprocess import call
from locofeature import *
from numpy import *
from subprocess import PIPE, run
import numpy as np
import cv2
import re

# image = cv2.imread('./LeafIdentity/LabSeg2/TrainSeg2/Acer/ny1010-02-1.jpg',0)

# def cProg(image):
def cfeats(image):
	c = UseLocoEfa(image)
	data = np.squeeze(c)

	np.savetxt('leafcontour.csv',data,delimiter = ',')

	#call(['./loco-efa_example.exe','./leafcontour.csv'])
	result = run(['./loco-efa_example.exe','./leafcontour.csv'], stdout=PIPE, stderr=PIPE, universal_newlines=True)

	coutput = result.stdout
	pattern = re.compile(r'(Ln\=)([0-9]+(\.[0-9]+)?)')

	newrow = []
	for (ln, value, decimal) in re.findall(pattern, coutput):
		newrow.append(float(value))
	return newrow