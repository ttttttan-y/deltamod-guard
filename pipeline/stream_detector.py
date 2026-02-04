import numpy as np

def ewma(x, alpha=0.2):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1,len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def online_threshold(series, z_k=3.0, warm=200):
    x = np.asarray(series, dtype=float)
    mu = np.mean(x[:warm]) if warm < len(x) else np.mean(x)
    sigma = np.std(x[:warm]) + 1e-9 if warm < len(x) else np.std(x) + 1e-9
    thr = mu + z_k * sigma
    return thr

def detect_on_overload(overload_ma, thr, cooldown=30):
    alerts = []
    last = -1e9
    for i, v in enumerate(overload_ma):
        if v >= thr and (i - last) >= cooldown:
            alerts.append(i); last = i
    return alerts
