from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys


def make_diagonal(x):
    """ Converts a vector into an diagonal matrix """
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2==0] = 1
    return X / np.expand_dims(l2,axis)


def polynomial_features():
    pass


def shuffle_data(X,y, seed=None):
    """ Random shuffle of the samples in X and y"""
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx],y[idx]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """Split the data into train and test sets"""
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # test_size
    split_i = len(y) - int(len(y) // (1/test_size))
    X_train, X_test = X[:split_i],X[split_i:]
    y_train, y_test = y[:split_i],y[split_i:]

    return X_train, X_test, y_train, y_test







