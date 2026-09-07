import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X, dtype = float)
    mean_matrix = np.mean(X, axis = 0)
    centered = X - mean_matrix
    return centered.T @ centered / (X.shape[0] - 1.0)