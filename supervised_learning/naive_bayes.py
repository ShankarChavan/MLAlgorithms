from __future__ import division, print_function
import numpy as np
import math


class  NaiveBayes():
    """The Gaussian naive bayes classifier""" 
    def fit(self, X, y):
        self.X,self.y=X,y
        self.classes=np.unique(y)
        self.parameters=[]
        # calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            # Only select the rows where labels equal the given classes
            X_where_c = X[np.where(y==c)]
            self.parameters.append([])
            # Add mean and variance for each feature(column)
            for col in X_where_c.T:
                parameters={"mean":col.mean(),"var":col.var()}
                self.parameters[i].append(parameters)
    
    def _calculate_likelihood(self, mean,var,x):
        """
        Gaussian likelihood of data given mean and var
        """
        eps=1e-4
        coeff=1.0 / math.sqrt(2.0*math.pi*var+eps)
        exponent=math.exp(-(math.pow(x-mean,2)/(2*var+eps)))
        return coeff * exponent
    
    def _calculate_prior(self, c):
        """calculate prior of class c"""
        freq = np.mean(self.y==c)
        return freq
    
    def _classify(self, sample):
        """Classification using bayes rule P(Y|X)=P(X|Y)*P(Y)/P(X),
           or Posterior = Likelihood * Prior / Scaling Factor
        """
        posteriors = []
        # Go through list of classes
        for i, c in enumerate(self.classes):
            # initialize posterior as prior
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(sample, self.parameters[i]):
                likelihood = self._calculate_likelihood(params["mean"],params["var"],feature_value)
                posterior*=likelihood
            posteriors.append(posterior)
        # Return the class with the largest posterior probability    
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        """
        Predict the class labels of the samples in X
        """
        y_pred = [self._classify(sample) for sample in X]
        return y_pred



