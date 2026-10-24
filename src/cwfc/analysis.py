from __future__ import annotations

import numpy as np

from .metrics import mae, normalized_mae   # starter functions, reused


def mase(y_true, y_pred, y_train) -> float:
    """MAE scaled by the in-sample one-step naive MAE of the training series."""
    scale = float(np.mean(np.abs(np.diff(np.asarray(y_train, dtype=float)))))
    err = mae(y_true, y_pred)
    if scale == 0:
        return 0.0 if err == 0 else float("inf")
    return err / scale


def rel_improvement(y_true, y_pred, baseline_pred) -> float:
    """Fraction by which the model's MAE beats the baseline's (positive = better)."""
    return 1.0 - normalized_mae(y_true, y_pred, baseline_pred)


def peak_mae(acts, preds, q: float = 0.9) -> float:
    """MAE restricted to the highest-load points (actual >= q-quantile)."""
    acts, preds = np.asarray(acts, float), np.asarray(preds, float)
    mask = acts >= np.quantile(acts, q)
    return mae(acts[mask], preds[mask])


def residual_band(errs, lo: float = 0.1, hi: float = 0.9, bucket: int = 6):
    """errs = actual - forecast, shape (folds, horizon), from the SELECTED model's
    rolling-origin run. Quantiles are pooled over buckets of horizon steps because
    a handful of folds per step is too few. Band = forecast + lower, forecast + upper."""
    H = errs.shape[1]
    lower, upper = np.empty(H), np.empty(H)
    for s in range(0, H, bucket):
        blk = errs[:, s:s + bucket].ravel()
        lower[s:s + bucket] = np.quantile(blk, lo)
        upper[s:s + bucket] = np.quantile(blk, hi)
    return lower, upper