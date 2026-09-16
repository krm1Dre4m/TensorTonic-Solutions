import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    y_pred = np.asarray(y_pred, dtype = float)
    y_true = np.asarray(y_true, dtype = float)
    return float(np.sum((y_pred - y_true) ** 2) / y_pred.shape[0])