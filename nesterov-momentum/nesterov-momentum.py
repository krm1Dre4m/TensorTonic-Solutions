import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    w = np.array(w, dtype = float)
    v = np.array(v, dtype = float)
    g = np.array(grad, dtype = float)
    
    new_v = momentum * v + lr * g
    new_w = w - new_v
    return {
        "new_w": new_w,
        "new_v": new_v
    }

    