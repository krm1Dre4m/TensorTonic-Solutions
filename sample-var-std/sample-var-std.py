import numpy as np

def sample_var_std(x: list) -> dict:
    x = np.asarray(x, dtype = float)
    var_list = np.var(x, ddof = 1)
    return {
        "variance": float(var_list),
        "standard_deviation": float(var_list ** 0.5)
    }