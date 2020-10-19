import numpy as np
import math
import sys

def calculate_entropy(y):
    """Calculate entropy of label array y"""
    log2 = lambda x:math.log(x)/math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y==label])
        p = count/len(y)
        entropy += -p*log2
    return entropy

def calculate_covariance_matrix(X, Y=None):
    """Calculate Covariance matrix for the dataset X"""
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1/(n_samples-1))*(X-X.mean(axis=0)).T.dot(Y-Y.mean(axis=0))

    return np.array(covariance_matrix,dtype=float)

def accuracy_score(y_true,y_pred):
    """Compare y_true to y_pred"""
    accuracy=np.sum(y_true==y_pred,axis=0)/len(y_true)
    return accuracy

def euclidean_distance(x1,x2):
    """calculate the l2 distance between two vectors"""
    distance=0
    #squared distance between each co-ordinate
    for i in range(len(x1)):
        distance += pow((x1[i]-x2[i]),2)
    return math.sqrt(distance)

