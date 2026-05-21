import math
def clip(y_pred, low, high):
    res = []
    for i in y_pred:
        i = float(i)
        if i < low:
            res.append(low)
        elif i > high:
            res.append(high)
        else: res.append(i)
    return res

def log_loss(y_true, y_pred, eps=1e-15):
    p = clip(y_pred, eps, 1 - eps)
    res = []
    for yt, yp  in zip(y_true, p):
        res.append(-(yt * math.log(yp) + (1 - yt) * math.log(1 - yp)))

    return res