from __future__ import annotations

import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)

import numpy as np

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

def ar1_meanrev(y, h):
    """Forecast = mean + phi^k * (last deviation). phi estimated on history only."""
    mu = y.mean()
    d = y - mu
    denom = np.dot(d[:-1], d[:-1])
    phi = np.clip(np.dot(d[1:], d[:-1]) / denom, 0, 0.99) if denom > 0 else 0.0
    return mu + phi ** np.arange(1, h + 1) * d[-1]

def seasonal_naive(m):
    def f(y, h):
        return np.resize(y[-m:], h)      # repeats the last m values
    f.__name__ = f"snaive{m}"
    return f

CANDIDATES = {
    "last_value": last_value,
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
}

def ar1_meanrev(y, h):
    """Forecast = mean + phi^k * (last deviation). phi estimated on history only."""
    mu = y.mean()
    d = y - mu
    denom = np.dot(d[:-1], d[:-1])
    phi = np.clip(np.dot(d[1:], d[:-1]) / denom, 0, 0.99) if denom > 0 else 0.0
    return mu + phi ** np.arange(1, h + 1) * d[-1]

def seasonal_naive(m):
    def f(y, h):
        return np.resize(y[-m:], h)      # repeats the last m values
    f.__name__ = f"snaive{m}"
    return f