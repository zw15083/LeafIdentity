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
		image = cv2.imread(input_path+folder+'/'+file,0)
		found_features = round(main(image),2)
		print(found_features)
		features.append(found_features)
		
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
		image = cv2.imread(input_path+folder+'/'+file,0)
		found_features = round(main(image),2)
		features2.append(found_features)
		
	fol += 1

# np.savetxt('features.csv',features,delimiter=',')
# np.savetxt('features2.csv',features2,delimiter=',')

# features = np.fromfile('features.csv',dtype = float,sep=',',count=-1)
# print(features)
# features2 = np.fromfile('features2.csv',dtype = float,sep=',',count=-1)

# features = features.resize([len(labels),1])
# features2 = features2.resize([len(labels2),1])


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
model = gnb.fit(features,labels)

preds = gnb.predict(features2)
print(preds)
print(accuracy_score(labels2, preds))