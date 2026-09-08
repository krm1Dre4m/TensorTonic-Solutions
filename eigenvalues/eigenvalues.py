import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    # Write code here
    X = np.asarray(matrix, dtype = float)
    
    eigenvals = np.linalg.eigvals(X).real
    return np.sort(eigenvals)
    