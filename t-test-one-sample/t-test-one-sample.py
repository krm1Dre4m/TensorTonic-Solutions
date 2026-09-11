import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    # Write code here
    A = np.asarray(x, dtype = float)
    n = A.shape[0]
    x_mean = np.mean(A)
    centered = A - x_mean
    
    std_dev = float(np.sqrt(np.sum(centered ** 2) / (n - 1)))
    if std_dev == 0:
        return 0.0 if x_mean == mu0 else float(np.inf)
    t = float((np.mean(A) - mu0) / (std_dev / np.sqrt(n)))
    return t