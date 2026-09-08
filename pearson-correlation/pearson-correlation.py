import numpy as np
    
def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X, dtype = float)
    centered = X - np.mean(X, axis = 0)

    cov = centered.T @ centered / (X.shape[0] - 1.0)
    std_dev = np.sqrt(np.diag(cov))
    outer_matrix = np.outer(std_dev, std_dev)

    return np.where(outer_matrix == 0, np.nan, cov / outer_matrix)