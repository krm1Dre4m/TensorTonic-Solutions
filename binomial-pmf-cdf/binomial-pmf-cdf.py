import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    pmf = float(math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k)))
    cdf = 0
    for i in range(0, k + 1):
        cdf += float(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)))
    return {
        "pmf": pmf,
        "cdf": cdf
    }