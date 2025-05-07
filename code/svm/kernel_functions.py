import numpy as np

def linear_kernel(X1, X2):
    """Compute the linear kernel between two matrices."""
    return np.dot(X1, X2.T)

def rbf_kernel(X1, X2, gamma=0.05):
    """Compute the RBF (Gaussian) kernel between two matrices."""
    if X1.ndim == 1:
        X1 = X1[np.newaxis, :]
    if X2.ndim == 1:
        X2 = X2[np.newaxis, :]
        
    dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dists)

def polynomial_kernel(X1, X2, degree=3, coef0=1):
    """Compute the polynomial kernel between two matrices."""
    return (np.dot(X1, X2.T) + coef0) ** degree
