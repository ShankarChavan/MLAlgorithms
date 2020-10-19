from __future__ import print_function, division
import numpy as np
import math


class Regression(object):
    """ Base regrssion model. 
    Models the relationship between a scalat dependent variable y 
    and independent variables x. 
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations=n_iterations
        self.learning_rate=learning_rate

    def initialize_weights(self, n_features):
        """Initialize weights randomly[-1/N, 1/N] """
        limit = 1/math.sqrt(n_features)
        self.w=np.random.uniform(-limit,limit,(n_featues,))
    
    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradiant descent for n_iteration
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            
            # calculate l2 loss
            mse = np.mean(0.5*(y - y_pred)**2 + self.regularization(self.w))

            self.training_errors.append(mse)
            
            # Gradient of l2 loss w.r.t w
            grad_w= -(y-y_pred).dot(X)+self.regularization(self.w)
            
            # update the weights
            self.w -= self.learning_rate*grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """ Linear model.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations, 
                                                learning_rate=learning_rate)
    
    def fit(self, X, y):
        # If not gradient descent ==> least square approximate  of w
        if not self.gradient_descent:
            # Insert constant ones for bias
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares(using Moore-Penrose pseudoinverse)
            U,S,V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression,self).fit(X,y)