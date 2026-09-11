import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    # Write code here
    A = np.asarray(A, dtype = float)
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1] or np.linalg.det(A) == 0:
        return None
    
    augmented = np.concatenate((A, np.identity(n)), axis = 1)
    for column in range(n):
        pivot = column + np.argmax(np.abs(augmented[column:, column]))
        if abs(augmented[pivot, column]) < 1e-12:
            return None
        augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]

        for row in range(n):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]

    return augmented[:, n:]
    
    
    