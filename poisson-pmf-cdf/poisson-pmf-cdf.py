import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    arr = np.arange(1, k + 1)
    pmf = float((lam ** k) / (np.exp(lam) * np.prod(arr)))

    cdf = 0.0
    fact = 1
    for i in range (0, k + 1):
        if i > 0:
            fact *= i
        cdf += float((lam ** i) / (np.exp(lam) * fact))
    
    return {
        "pmf": pmf,
        "cdf": cdf
    }
    