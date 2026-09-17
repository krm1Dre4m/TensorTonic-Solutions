import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.array(param)
    g = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * (g ** 2)

    m_bc = m_new / (1 - beta1 ** t)
    v_bc = v_new / (1 - beta2 ** t)

    p_new = p - lr * m_bc / (v_bc ** 0.5 + eps)

    return p_new, m_new, v_new