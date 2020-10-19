from __future__ import print_function, division
import numpy as np
import math
from utils.activation_functions import Sigmoid
from utils.feature_manipulation import make_diagonal

class LogisticRegression():
    """ Logistic Regression classifier

    """
    def __init__(self,learning_rate=.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid=Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        limit = 1/math.sqrt(n_features)
        self.param=np.random.uniform(-limit,limit,(n_features,))
    
    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # make new prediction
            y_pred=self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                # move against the gradient of loss function w.r.t 
                # parameters to minimize the loss
                self.param-=self.learning_rate*-(y-y_pred).dot(X)
            else:
                diag_gradient=make_diagonal(self.sigmoid.gradient(X.dot(self.param)))

                self.param=np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param)+y - y_pred)
    
    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

