import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def baseline_arima_resid_detect(x, order=(2,1,2), z_k=3.0, warm=200):
    x = np.asarray(x, dtype=float)
    try:
        res = ARIMA(x, order=order).fit()
        resid = res.resid
    except Exception:
        resid = x - np.mean(x)
    mu = np.mean(resid[:warm]); sigma=np.std(resid[:warm])+1e-9
    thr = mu + z_k*sigma
    alerts = [i for i,r in enumerate(resid) if r>=thr]
    return resid, alerts

def baseline_iforest(x, cont=0.01):
    x = np.asarray(x, dtype=float).reshape(-1,1)
    clf = IsolationForest(contamination=cont, random_state=0).fit(x)
    score = -clf.decision_function(x)  # larger = more abnormal
    thr = np.quantile(score, 0.99)
    alerts = [i for i,v in enumerate(score) if v>=thr]
    return score, alerts

def baseline_ocsvm(x, nu=0.01, gamma='scale'):
    x = np.asarray(x, dtype=float).reshape(-1,1)
    clf = OneClassSVM(nu=nu, gamma=gamma).fit(x)
    score = -clf.decision_function(x)
    thr = np.quantile(score, 0.99)
    alerts = [i for i,v in enumerate(score) if v>=thr]
    return score, alerts
