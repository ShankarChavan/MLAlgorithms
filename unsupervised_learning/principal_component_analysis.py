import numpy as np
from utils.feature_operation import calculate_covariance_matrix

class PCA():
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features 
    and maximizing the variance along each feature axis.This class is also used throughout
    the project to plot data.
    """
    def transform(self, X, n_components):
        covariance_matrix = calculate_covariance_matrix(X)
        eigenvalues,eigenvectors = np.linalg.eig(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues=eigenvalues[idx][:n_components]
        eigenvectors=np.atleast_1d(eigenvectors[:,idx])[:,:n_components]

        #project data into principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed