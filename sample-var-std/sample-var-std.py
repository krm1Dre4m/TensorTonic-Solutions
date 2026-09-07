import numpy as np

def sample_var_std(x: list) -> dict:
    x = np.asarray(x, dtype = float)
    mean_arr = np.mean(x)
    centered = x - mean_arr
    var_arr = float(np.sum(centered ** 2)) / (np.size(centered) - 1)
    std_dev = float(var_arr ** 0.5)
    return {
        "variance": var_arr,
        "standard_deviation": std_dev
    }