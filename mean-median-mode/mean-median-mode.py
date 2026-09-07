from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    if not x:
        raise ValueError("Input list cannot be empty.")

    x_arr = np.asarray(x)
    
    mean_val = float(np.mean(x_arr))
    median_val = float(np.median(x_arr))
    
    counts = Counter(x)
    max_freq = max(counts.values())
    
    mode_val = min(val for val, freq in counts.items() if freq == max_freq)

    return {
        "mean": mean_val,
        "median": median_val,
        "mode": float(mode_val)
    }