import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    # Write code here
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype = float)
    n = x.shape[0]
    bootstrap_mean = []
    for _ in range (0, n_bootstrap):
        sample = rng.choice(x, size = n, replace = True)
        bootstrap_mean.append(float(np.mean(sample)))
    alpha = (1 - ci) / 2
    return{
        "bootstrap_mean": float(np.mean(bootstrap_mean)),
        "lower": float(np.quantile(bootstrap_mean, alpha)),
        "upper": float(np.quantile(bootstrap_mean, 1 - alpha))
    }
