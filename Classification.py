from numpy import *
import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import feature1 as f1
import feature3 as f3
from fourier_funk import contour
from fourier_funk import bounding_box
from locofeature import *
from runC import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

label_names = ['Acer','Aesculus','Betula','Carya','Cornus','Fraxinus',
				'Magnolia','Malus','Pinus','Populus','Prunus','Quercus','Salix']
				
				
def get_cfeature():
	input_path = 'LabSeg2/TrainSeg2/'
	folders = next(os.walk(input_path))[1]
	fol = 0
	labels = []
	features = []
	for folder in folders:
		files = next(os.walk(input_path+folder))[2]
		print(folder)
		for file in files:
			#here you would add in the values for each feature for a single leaf
			image = cv2.imread(input_path+folder+'/'+file,0)
			row = cfeats(image)
			features.append(row)
		fol += 1
		
	input_path2 = 'LabSeg2/TestSeg2/'
	folders = next(os.walk(input_path2))[1]
	fol = 0
	features2 = []
	labels2 = []
	for folder in folders:
		files = next(os.walk(input_path2+folder))[2]
		print(folder)
		for file in files:
			#here you would add in the values for each feature for a single leaf
			image = cv2.imread(input_path2+folder+'/'+file,0)
			row = cfeats(image)
			features2.append(row)

			
		fol += 1

	#print(features)
	np.savetxt('cfeature.txt',features)
	np.savetxt('cfeature2.txt',features2)
				
def get_feature():

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
			#here you would add in the values for each feature for a single leaf
			image = cv2.imread(input_path+folder+'/'+file,0)
			# [max,cont] = contour(image)
			# found_features = bounding_box(max)
			found_features = round(f3.main(image),2)
			# found_features = round(f1.main(image),2)
			features.append(found_features)
			
		fol += 1
		
	input_path2 = 'LabSeg2/TestSeg2/'
	folders = next(os.walk(input_path2))[1]
	fol = 0
	features2 = []
	labels2 = []
	for folder in folders:
		files = next(os.walk(input_path2+folder))[2]
		print(folder)
		for file in files:
			#here you would add in the values for each feature for a single leaf
			image = cv2.imread(input_path2+folder+'/'+file,0)
			# [max,cont] = contour(image)
			# found_features = bounding_box(max)
			found_features = round(f3.main(image),2)
			# found_features = round(f1.main(image),2)
			features2.append(found_features)
			
		fol += 1

	#print(features)
	np.savetxt('npeaks.txt',features)
	np.savetxt('npeaks2.txt',features2)
	
def get_label():
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
			
		fol += 1
		
	input_path2 = 'LabSeg2/TestSeg2/'
	folders = next(os.walk(input_path2))[1]
	fol = 0
	features2 = []
	labels2 = []
	for folder in folders:
		files = next(os.walk(input_path2+folder))[2]
		print(folder)
		for file in files:
			labels2.append(fol)
			
		fol += 1
	return [labels,labels2]

#get_feature()
#get_cfeature()
box1 = np.loadtxt('./boxfeature.txt')
box2 = np.loadtxt('./boxfeature2.txt')
npeaks1 = np.loadtxt('./npeaks.txt')
npeaks2 = np.loadtxt('./npeaks2.txt')
tbox1 = np.loadtxt('./tbox.txt')
tbox2 = np.loadtxt('./tbox2.txt')
cfeature = np.loadtxt('./cfeature.txt')
cfeature2 = np.loadtxt('./cfeature2.txt')
[labels,labels2] = get_label()
# print(labels)

features11 = np.reshape(tbox1,(shape(labels)[0],1))
features12 = np.reshape(tbox2,(shape(labels2)[0],1))
features21 = np.reshape(box1,(shape(labels)[0],1))
features22 = np.reshape(box2,(shape(labels2)[0],1))
feature31 = np.reshape(npeaks1,(shape(labels)[0],1))
feature32 = np.reshape(npeaks2,(shape(labels2)[0],1))
featureTrain = column_stack((features11,features21,feature31,cfeature))
featureTest = column_stack((features12,features22,feature32,cfeature2))
print(shape(featureTrain))
print(shape(featureTest))

# gnb = GaussianNB()
# model = gnb.fit(feature1,labels)

# preds = gnb.predict(feature2)
# print(preds)
# print(accuracy_score(labels2, preds))

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(featureTrain,labels)
preds = clf.predict(featureTest)
print(preds)
print(accuracy_score(labels2, preds))