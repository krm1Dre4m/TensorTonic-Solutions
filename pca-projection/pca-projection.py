import numpy as np

def pca_projection(X: list, k: int) -> list:
    # centering the data
    X = np.asarray(X, dtype = float)
    n = X.shape[0]
    centered = X - np.mean(X, axis = 0)
    cov_mav = centered.T @ centered / (n - 1)

    # top k eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_mav)
    sorted_eigvals = np.argsort(eigvals)[::-1]
    top_eigvals = sorted_eigvals[:k]
    top_eigvecs = eigvecs[:, top_eigvals]

    return centered @ top_eigvecs
    