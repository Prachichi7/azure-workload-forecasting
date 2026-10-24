from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import Ridge


def last_value(y, h):
    return np.full(h, y[-1])


def moving_average(k):
    def f(y, h):
        return np.full(h, np.mean(y[-k:]))
    f.__name__ = f"ma{k}"
    return f


def expanding_mean(y, h):
    return np.full(h, np.mean(y))


def ewma(alpha):
    def f(y, h):
        level = y[0]
        for v in y[1:]:
            level = alpha * v + (1 - alpha) * level
        return np.full(h, level)
    f.__name__ = f"ewma{alpha}"
    return f


def drift(y, h):
    k = (y[-1] - y[0]) / max(len(y) - 1, 1)
    return y[-1] + k * np.arange(1, h + 1)


def seasonal_naive(m):
    def f(y, h):
        return np.resize(y[-m:], h)      # repeats the last m values
    f.__name__ = f"snaive{m}"
    return f


def ar1_meanrev(y, h):
    """Forecast = mean + phi^k * (last deviation). phi estimated on history only."""
    mu = y.mean()
    d = y - mu
    denom = np.dot(d[:-1], d[:-1])
    phi = np.clip(np.dot(d[1:], d[:-1]) / denom, 0, 0.99) if denom > 0 else 0.0
    return mu + phi ** np.arange(1, h + 1) * d[-1]


def direct_ar1(y, h):
    """Per-horizon regression of (y[t+k]-mu) on (y[t]-mu), fit on history only.
    Each horizon gets its own persistence: step 1 can lean on the last value,
    step 24 reverts to the mean."""
    mu = y.mean()
    d = y - mu
    out = np.empty(h)
    for k in range(1, h + 1):
        x, t = d[:-k], d[k:]
        denom = float(x @ x) if len(x) else 0.0
        beta = np.clip((x @ t) / denom, 0, 1) if (len(x) >= 5 and denom > 0) else 0.0
        out[k - 1] = mu + beta * d[-1]
    return out


def arima(order):
    def f(y, h):
        from statsmodels.tsa.arima.model import ARIMA
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = np.asarray(ARIMA(y, order=order).fit().forecast(h))
            if np.isfinite(fc).all():
                return fc
        except Exception:
            pass
        return np.full(h, y.mean())          # documented fallback
    f.__name__ = "arima" + "".join(map(str, order))
    return f


def ridge_direct(alpha=10.0, lags=(0, 1, 2, 5, 11)):
    """One ridge model per horizon step. Features at origin t use only d[:t+1]."""
    L = max(lags)

    def feat(d, t):
        return [d[t - j] for j in lags] + [d[t - 5:t + 1].mean()]

    def f(y, h):
        mu = y.mean()
        d = y - mu
        n = len(d)
        out = np.full(h, mu)
        if n <= L:
            return out
        x_now = np.array(feat(d, n - 1)).reshape(1, -1)
        for k in range(1, h + 1):
            ts = range(L, n - k)
            if len(ts) < 15:
                continue
            X = np.array([feat(d, t) for t in ts])
            Y = np.array([d[t + k] for t in ts])
            out[k - 1] = mu + Ridge(alpha=alpha).fit(X, Y).predict(x_now)[0]
        return out

    f.__name__ = f"ridge{alpha}"
    return f


CANDIDATES = {
    "last_value": last_value,        # first = wins ties (e.g. constant active_vm_count)
    "ma6": moving_average(6),
    "ma12": moving_average(12),
    "ma24": moving_average(24),
    "expanding_mean": expanding_mean,
    "ewma0.1": ewma(0.1),
    "ewma0.3": ewma(0.3),
    "drift": drift,
    "ar1_meanrev": ar1_meanrev,
    "snaive12": seasonal_naive(12),
    "snaive24": seasonal_naive(24),
    "direct_ar1": direct_ar1,
    "arima101": arima((1, 0, 1)),
    "ridge10": ridge_direct(10.0),
}