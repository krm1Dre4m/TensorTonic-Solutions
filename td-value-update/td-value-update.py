import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    V = np.array(V, dtype = float)
    V[s] += alpha * (r + gamma * V[s_next] - V[s])
    return V