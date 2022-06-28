# -*- coding: utf-8 -*-
"""
Daniel Carpenter

Machine Learning
Testing Performance of Models to Read Hand Written Notes
Python for DSA
"""



# Import packages
import numpy as np
import pandas as pd
import sklearn


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn import datasets, metrics
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split


# Get data - see more info from here https://www.openml.org/d/554
# Data consists of hand written letters 
digits = datasets.load_digits()


# Data Transformation
X = digits.data / 255 # normalize between 0 and 1 - can improve performance
y = digits.target

percSampleToUse = 0.66 # Percentage of sample to use in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=percSampleToUse, 
                                                    random_state=9999)


# Bayes -----------------------------------------------------------------------

# 0 - Import the classifer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Initialize the classifier
gnb = GaussianNB()

# 2. Train the classifier
model = gnb.fit(X_train, y_train)

# 3. Make predictions
preds = gnb.predict(X_test)

# 4. Evaluate accuracy
print('\nNaive Bayes Accuracy Score: {:,.4f}'.format( accuracy_score(y_test, preds)) )

# 5. Interprest false and true positives
expected  = y_test
predicted = preds
print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(expected, predicted))


# =============================================================================
# Random Forest 
# =============================================================================

# 0 - Import the classifer
from sklearn.ensemble import RandomForestClassifier 

# 1. Initialize the classifier
forest = RandomForestClassifier(n_estimators = 100)

# 2. Train the classifier
forest.fit(X_train, y_train)

# 3. Make predictions
predictions = forest.predict(X_test)

# 4. Evaluate accuracy
print('\nRandom Forest Accuracy Score: {:,.4f}'.format( forest.score(X_test, y_test)) )

# 5. Interprest false and true positives
expected  = y_test
predicted = predictions
print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(expected, predicted))


