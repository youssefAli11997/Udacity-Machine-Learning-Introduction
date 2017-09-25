#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# if we want to use just 1% of the training set
'''features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]'''

#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)
t0 = time()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Training and prediction time:", (time()-t0), "sec."
print len(pred[pred == 0]), "emails predicted to be written by Sally"
print len(pred[pred == 1]), "emails predicted to be written by Chris"
print "prediction[10] =", pred[10]
print "prediction[26] =", pred[26]
print "prediction[50] =", pred[50]
print clf.score(features_test, labels_test)
#########################################################


