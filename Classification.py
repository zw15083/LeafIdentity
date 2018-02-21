from numpy import *
import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from feature1 import main

label_names = ['Acer','Aesculus','Betula','Carya','Cornus','Fraxinus',
				'Magnolia','Malus','Pinus','Populus','Prunus','Quercus','Salix']

#it is possible to get this to seperate into test/training data
input_path = 'LabSeg2/TrainSeg2/'
folders = next(os.walk(input_path))[1]
fol = 0
labels = []
features = []
for folder in folders:
	files = next(os.walk(input_path+folder))[2]
	print(folder)
	for file in files:
		labels.append(fol)
		#here you would add in the values for each feature for a single leaf
		# image = cv2.imread(input_path+folder+'/'+file,0)
		# found_features = round(main(image),2)
		# features.append(found_features)
		
	fol += 1
	
input_path2 = 'LabSeg2/TestSeg2/'
folders = next(os.walk(input_path))[1]
fol = 0
features2 = []
labels2 = []
for folder in folders:
	files = next(os.walk(input_path+folder))[2]
	print(folder)
	for file in files:
		labels2.append(fol)
		#here you would add in the values for each feature for a single leaf
		# image = cv2.imread(input_path+folder+'/'+file,0)
		# found_features = round(main(image),2)
		# features2.append(found_features)
		
	fol += 1

# np.savetxt('features.txt',features)
# np.savetxt('features2.txt',features2)

features = np.loadtxt('./features.txt')
features2 = np.loadtxt('./features2.txt')


features = np.reshape(features2,(shape(labels2)[0],1))
features2 = np.reshape(features2,(shape(labels2)[0],1))





gnb = GaussianNB()
model = gnb.fit(features,labels)

preds = gnb.predict(features2)
print(preds)
print(accuracy_score(labels2, preds))