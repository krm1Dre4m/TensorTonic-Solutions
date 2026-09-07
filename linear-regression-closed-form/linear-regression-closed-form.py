import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Write code here
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)
    return np.linalg.inv(X.T @ X) @ X.T @ y