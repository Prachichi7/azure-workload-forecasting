from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out

import numpy as np

def rolling_origins(n, horizon, min_train, step):
    """History y[:o], test window y[o:o+horizon]. The last origin is n - horizon,
    so the final window ends exactly at n (the end of the TRAINING window)."""
    first = max(min_train, 2)
    last = n - horizon
    if last < first:
        return []
    return list(range(last, first - 1, -step))[::-1]

def rolling_origin_eval(y, model_fn, horizon=24, min_train=48, step=6):
    """y = TRAINING window only. Returns (errors, actuals) arrays of shape (folds, horizon)."""
    y = np.asarray(y, float)
    errs, acts = [], []
    for o in rolling_origins(len(y), horizon, min_train, step):
        pred = model_fn(y[:o], horizon)
        act = y[o:o + horizon]
        errs.append(act - pred)
        acts.append(act)
    return np.array(errs), np.array(acts)

def summarize(errs):
    return {"mae": float(np.mean(np.abs(errs))),
            "rmse": float(np.sqrt(np.mean(errs ** 2)))}