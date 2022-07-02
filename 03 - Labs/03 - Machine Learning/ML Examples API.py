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


from sklearn.datasets import fetch_openml
from sklearn import datasets, metrics
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split


# =============================================================================
# GET DATA
# he MNIST dataset consists of 70,000 handwritten numeric digits used to evaluate algorithms for classification
# See more info from here https://www.openml.org/d/554
# =============================================================================

# Data consists of hand written numbers 
digits = datasets.load_digits()

# Data Transformation
X = digits.data / 255 # normalize between 0 and 1 - can improve performance

y = digits.target

# Percentage of sample to use in the test set
percSampleToUse = 0.66 

# Make the training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=percSampleToUse, 
                                                    random_state=9999)


# =============================================================================
# Initial Data Exploration
# =============================================================================

# Simple function to show some handwritten numeric observations

def showNumber(df, numberObsNum = 0):
        
    # What does one sample number look like in the data?
    # -----------------------------------------------------------------------------
    sampleNumber = df[numberObsNum][:]
    
    NUM_ROWS_COLS = 8
    sampleOut = np.reshape(sampleNumber, (NUM_ROWS_COLS, NUM_ROWS_COLS))
    
    print('Here is what the number looks like with raw data (After putting in a 8x8 matrix): \n\n',
          sampleOut)
    
    
    # How does it look like in a heat map perspective?
    # -----------------------------------------------------------------------------
    
    # Convert to DataFrame
    df_SampleNumber = pd.DataFrame(sampleOut)
    
    # Show a heatmap of the data
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create the plot and heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(df_SampleNumber, cmap="Blues")
    
    # Add some styles and guides
    sns.set_style("whitegrid")
    plt.title('How the handwritten number appears in data\n One handwritten numeric observation of the MNIST Dataset\n')
    plt.legend(title = 'Legend:\nMark Intensity')
    
    # The Axis'
    ax.set(ylabel = '"Rows" of Data')
    ax.set(xlabel = '"Columns" of Data')
    
    # Print and Show
    print('Note how the data looks like when putting it in a heatmap\n\n' )
    plt.show()


# Possible number: 2
showNumber(df=X, numberObsNum=50)

# Possible number: 9
showNumber(df=X, numberObsNum=1616)

# Possible number: 3
showNumber(df=X, numberObsNum=1300)

# Possible number: Another 3
showNumber(df=X, numberObsNum=999)

# Possible number: 6
showNumber(df=X, numberObsNum=212)


# =============================================================================
# Fit Classifier Models
# =============================================================================

# 1 - Naive Bayes 
# -----------------------------------------------------------------------------

# 0 - Import the classifer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Initialize the classifier
gnb = GaussianNB()

# 2. Train (fit) the classifier
model = gnb.fit(X_train, y_train)

# 3. Make predictions
preds = gnb.predict(X_test)

# 4. Evaluate accuracy
print('\nNaive Bayes Training Score: {:,.4f}'.format( gnb.score(X_train, y_train) ))
print('Naive Bayes Test Score: {:,.4f}'.format( gnb.score(X_test, y_test) ))
# 5. Interprest false and true positives
expected  = y_test
predicted = preds
print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(expected, predicted))


# 2 - Random Forest 
# -----------------------------------------------------------------------------

# 0 - Import the classifer
from sklearn.ensemble import RandomForestClassifier 

# 1. Initialize the classifier
forest = RandomForestClassifier(n_estimators = 100)

# 2. Train (fit) the classifier
forest.fit(X_train, y_train)

# 3. Make predictions
predictions = forest.predict(X_test)

# 4. Evaluate accuracy
print('\nRandom Forest Training Score: {:,.4f}'.format( forest.score(X_train, y_train) ))
print('Random Forest Test Score: {:,.4f}'.format( forest.score(X_test, y_test) ))

# 5. Interprest false and true positives
expected  = y_test
predicted = predictions
print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(expected, predicted))


# 3 - K-Nearest Neighbors
# -----------------------------------------------------------------------------

# 0 - Import the classifer
from sklearn.neighbors import KNeighborsClassifier

# 1. Initialize the classifier
neigh = KNeighborsClassifier(n_neighbors=3)

# 2. Train (fit) the classifier (3. predictions made in this step too)
neigh.fit(X, y)

# 4. Evaluate accuracy - note does test and train at same time (hence greedy)
print('\nKNN Accuracy Score: {:,.4f}'.format( neigh.score(X, y) ))

# Perception using SGD
# ‘sgd’ refers to stochastic gradient descent.
# -----------------------------------------------------------------------------
# 0 - Import the classifer
from sklearn.neural_network import MLPClassifier

# 1. Initialize the classifier
mlp_sgd = MLPClassifier(hidden_layer_sizes=(100,), 
                    max_iter=100, 
                    alpha=1e-4, 
                    solver="sgd",
                    verbose = 10, 
                    tol=1e-4, 
                    random_state=1, 
                    learning_rate_init=.1)

# 2. Train (fit) the classifier 
mlp_sgd.fit(X_train, y_train)

# 3. Make predictions
predictions = mlp_sgd.predict(X_test)

# 4. Evaluate accuracy
print('\nMLP with SGD Training Score: {:,.4f}'.format( mlp_sgd.score(X_train, y_train) ))
print('MLP with SGD Test Score: {:,.4f}'.format( mlp_sgd.score(X_test, y_test) ))

# 5. Interprest false and true positives
expected  = y_test
predicted = predictions
print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(expected, predicted))


# Perception using Adam
# ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
# -----------------------------------------------------------------------------
# 0 - Import the classifer
# from sklearn.neural_network import MLPClassifier # done in last MLP

# 1. Initialize the classifier
mlp_lbfgs = MLPClassifier(hidden_layer_sizes=(100,), 
                    max_iter=100, 
                    alpha=1e-4, 
                    solver="adam",
                    verbose = 10, 
                    tol=1e-4, 
                    random_state=1, 
                    learning_rate_init=.1)

# 2. Train (fit) the classifier 
mlp_lbfgs.fit(X_train, y_train)

# 3. Make predictions
predictions = mlp_lbfgs.predict(X_test)

# 4. Evaluate accuracy
print('\nMLP with Adam Training Score: {:,.4f}'.format( mlp_lbfgs.score(X_train, y_train) ))
print('MLP with Adam Test Score: {:,.4f}'.format( mlp_lbfgs.score(X_test, y_test) ))

# 5. Interprest false and true positives
expected  = y_test
predicted = predictions
print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(expected, predicted))


# 4 - SVM (Support Vector Machine)
# https://scikit-learn.org/stable/modules/svm.html#classification
# -----------------------------------------------------------------------------

# 0 - Import the classifer
from sklearn import svm

# 1. Initialize the classifier
svm_model = svm.SVC()

# 2. Train (fit) the classifier 
svm_model.fit(X_train, y_train)

# 3. Make predictions
predictions = svm_model.predict(X_test)

# 4. Evaluate accuracy
print('\nSVM Training Score: {:,.4f}'.format( svm_model.score(X_train, y_train) ))
print('SVM Test Score: {:,.4f}'.format( svm_model.score(X_test, y_test) ))

# 5. Interprest false and true positives
expected  = y_test
predicted = predictions
print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(expected, predicted))


# 5 - Bayesian Ridge Regression
# https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression# -----------------------------------------------------------------------------

# 0 - Import the classifer
from sklearn import linear_model

# 1. Initialize the classifier
bayes_ridgeReg = linear_model.BayesianRidge()

# 2. Train (fit) the classifier 
bayes_ridgeReg.fit(X_train, y_train)

# 3. Make predictions
predictions = bayes_ridgeReg.predict(X_test)

# 4. Evaluate accuracy
print('\nBayesian Ridge Reg. Training Score: {:,.4f}'.format( bayes_ridgeReg.score(X_train, y_train) ))
print('Bayesian Ridge Reg. Test Score: {:,.4f}'.format( bayes_ridgeReg.score(X_test, y_test) ))


# TODO - Need to do visuals

# Summary of Classifier Models
# -----------------------------------------------------------------------------
# 1. Which model is the most accurate on the test set?
# The best model was the support vector machine model (SVM), which had a test score of 0.967

# 2. Which model is the fastest to train? # TODO

# 3. Which model is the fastest to classify new data (the test set)? # TODO

# 4. Which model is the best?
# The SVM model proves to predict out of sample and sample data very well, which looks to be the best overall
# The random forest model may have been better, but it appears to overfit the training data, which could be an issue for out of sample prediction.
