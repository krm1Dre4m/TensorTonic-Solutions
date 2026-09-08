import numpy as np

def covariance(X: np.ndarray, centered: np.ndarray) -> np.ndarray:
    return centered.T @ centered / (X.shape[0] - 1.0)

def standard_deviation(X: np.ndarray, centered: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(centered ** 2, axis = 0) / (X.shape[0] - 1.0))
    
def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X, dtype = float)
    centered = X - np.mean(X, axis = 0)

    cov = covariance(X, centered)
    std_dev = standard_deviation(X, centered)
    outer_matrix = np.outer(std_dev, std_dev)

    return np.where(outer_matrix == 0, np.nan, cov / outer_matrix)