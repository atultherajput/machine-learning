#!/usr/bin/env python3

"""

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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
### code goes here ###

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print ("Training time:", round(time()-t0, 3), "sec")

t0 = time()
pred = clf.predict(features_test)
print ("Prediction time:", round(time()-t0, 3), "sec")

from sklearn.metrics import accuracy_score
print ("Accuracy:", accuracy_score(pred, labels_test))

#########################################################


