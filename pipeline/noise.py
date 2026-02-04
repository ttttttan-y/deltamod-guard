import numpy as np

def synth_series(n, trend=None, season=None, noise_sigma=0.1):
    t = np.arange(n)
    y = np.zeros(n)
    if trend: y += trend.get("slope",0.0)*t
    if season and season.get("type")=="sin":
        per = season.get("period",128); amp = season.get("amp",1.0)
        y += amp*np.sin(2*np.pi*t/per)
    y += noise_sigma*np.random.randn(n)
    return y

def inject_anomalies(x, anomalies):
    """Return y, gt_idx (list of start indices)"""
    y = x.copy()
    gts = []
    for a in anomalies or []:
        t0 = int(a.get("t0",0)); L = int(a.get("length",1))
        typ = a.get("type","step")
        if typ=="step":
            amp = float(a.get("amp",1.0))
            y[t0:t0+L] += amp
            gts.append(t0)
        elif typ=="spike":
            amp = float(a.get("amp",3.0))
            y[t0] += amp
            gts.append(t0)
        elif typ=="drift":
            slope = float(a.get("slope",0.01))
            for k in range(L):
                idx = t0+k
                if idx<len(y): y[idx] += slope*k
            gts.append(t0)
    return y, sorted(list(set(gts)))
