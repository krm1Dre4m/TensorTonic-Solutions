import numpy as np

def entropy_node(y: list[int]) -> float:
    y = np.asarray(y, dtype = float)

    _, cnt = np.unique(y, return_counts = True)

    prob = np.where(len(y) == 0.0, 0.0, cnt / len(y))

    return float( -np.sum(prob * np.log2(prob)))