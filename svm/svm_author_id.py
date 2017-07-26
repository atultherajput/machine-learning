#!/usr/bin/env python3

"""
    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../libraries/tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000)

#features_train = features_train[:len(features_train)//100] 	#for linear kernel run fast
#labels_train = labels_train[:len(labels_train)//100]       		#for linear kernel run fast

t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
print ("accuracy:", accuracy_score(pred, labels_test))

print(pred[10],pred[26],pred[50]) #predict class (0 and 1) for element 10,26,50 of the test set

count = 0
for x in range(len(pred)):
    if pred[x] == 1:
        count += 1
print(count)    #predicted test event to be in the (1) class.
#########################################################


