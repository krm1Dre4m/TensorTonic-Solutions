import numpy as np

def dot_product(x: list, y: list) -> float:
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)
    return float(np.dot(x, y))