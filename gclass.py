from numpy import *
import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

	
label_names = ['Aesculus','Ficus','Gymocladus','Quercus']

a = np.loadtxt('./aesculus_mag.txt')
f = np.loadtxt('./ficus_mag.txt')
g = np.loadtxt('./gymocladus_mag.txt')
q = np.loadtxt('./quercus_mag.txt')
features = np.concatenate((a,f,g,q))
a0 = zeros(a.shape[0])
f1 = ones(f.shape[0])
g2 = ones(g.shape[0])*2
q3 = ones(q.shape[0])*3
labels = np.concatenate((a0,f1,g2,q3))
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.2,
                                                          random_state=42)


														  
gnb = GaussianNB()
model = gnb.fit(train,train_labels)

preds = gnb.predict(test)
print(preds)
print(accuracy_score(test_labels, preds))