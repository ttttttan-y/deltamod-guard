import numpy as np

def dm_encode(x, delta0=0.5, gamma=1.2, theta_k=1.0):
    """Delta Modulation / ADPCM-lite encoder.
    Returns:
      s: sign bits in {-1, +1}
      xhat: reconstructed signal
      delta: step size trajectory
      overload: 1 if |e_t| > theta_k * delta_t
      resid: reconstruction error
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = np.zeros(n)
    xhat = np.zeros(n)
    delta = np.zeros(n)
    overload = np.zeros(n)
    e = np.zeros(n)
    # init
    xhat[0] = x[0]
    delta[0] = delta0
    s[0] = 0
    for t in range(1, n):
        e[t] = x[t] - xhat[t-1]
        st = 1.0 if e[t] >= 0 else -1.0
        s[t] = st
        # overload check
        overload[t] = 1.0 if abs(e[t]) > theta_k * delta[t-1] else 0.0
        # adaptive step
        if overload[t] >= 1.0:
            dnext = delta[t-1] * gamma
        else:
            dnext = max(delta[t-1] / np.sqrt(gamma), 1e-6)
        delta[t] = dnext
        # reconstruct
        xhat[t] = xhat[t-1] + dnext * st
    resid = x - xhat
    return s, xhat, delta, overload, resid

def flip_rate(s, win=64):
    s = np.asarray(s, dtype=float)
    flips = np.r_[0, (np.sign(s[1:])-np.sign(s[:-1])!=0).astype(float)]
    # moving average
    k = win
    if k<=1: return flips
    c = np.convolve(flips, np.ones(k)/k, mode='same')
    return c

def run_lengths(s):
    """Run-lengths of consecutive identical signs."""
    s = np.asarray(s, dtype=float)
    if len(s)==0: return []
    runs = []
    cur = s[0]
    cnt = 1
    for v in s[1:]:
        if v==cur: cnt+=1
        else:
            runs.append(cnt); cnt=1; cur=v
    runs.append(cnt)
    return runs
