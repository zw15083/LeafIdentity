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
labels = []
#it is possible to get this to seperate into test/training data
input_path = '../LabSeg2/TrainSeg2/'  
folders = next(os.walk(input_path))[1]
fol = 0
features = []
for folder in folders:
	files = next(os.walk(input_path+folder))[2]
	for file in files:
		labels.append(fol)
		#here you would add in the values for each feature for a single leaf
		image = cv2.imread(input_path+folder+'/'+file,0)
		found_features = main(image)
		#features.append(found_features)
		features.append(found_features)
	fol += 1
	
print(shape(labels))
print(shape(features))
	
# label_names = ['ficus','quercus']
# q = np.loadtxt('./kernel_quercus5.txt')
# f = np.loadtxt('./kernel_ficus5.txt')
# features = np.concatenate((f, q))
# zero = zeros(f.shape[0])
# one = ones(q.shape[0])
# labels = np.concatenate((zero, one))
# train, test, train_labels, test_labels = train_test_split(features,
                                                          # labels,
                                                          # test_size=0.33,
                                                          # random_state=42)


gnb = GaussianNB()
model = gnb.fit(train,train_labels)

preds = gnb.predict(test)
print(preds)
print(accuracy_score(test_labels, preds))