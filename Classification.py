from numpy import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

label_names = ['Acer','Aesculus','Betula','Carya','Cornus','Fraxinus',
				'Magnolia','Malus','Pinus','Populus','Prunus','Quercus','Salix']
labels = []

input_path = './Lab/Train/'  
folders = next(os.walk(input_path))[1]
fol = 0
features = []
for folder in folders:
	files = next(os.walk(input_path+folder))[2]
	for file in files:
		labels.append(fol)
		#here you would add in the values for each feature for a single leaf
		found_features = []
		features.append(found_features)
	fol += 1
	


# gnb = GaussianNB()
# model = gnb.fit(features,labels)

# preds = gnb.predict(test)
# print(preds)
# print(accuracy_score(test_labels, preds))