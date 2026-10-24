from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .candidates import CANDIDATES
from .rolling import rolling_origin_eval, summarize
from .submission import validate_predictions
from .targets import HOLDOUT, TARGETS, build_targets, load_readings, split


def pick_model(y, horizon, min_train=48, step=6, tol=0.02):
    """Lowest rolling-origin RMSE on the training window; among candidates within
    `tol` of the best, the earliest (simplest) in CANDIDATES wins. Ties -> last_value."""
    scores = {}
    for name, f in CANDIDATES.items():
        errs, _ = rolling_origin_eval(y, f, horizon, min_train, step)
        if errs.size == 0:
            return "last_value"
        scores[name] = summarize(errs)["rmse"]
    best = min(scores.values())
    return next(n for n, s in scores.items() if s <= best * (1 + tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="predictions.csv")
    a = ap.parse_args()

    series = build_targets(load_readings(a.data))
    train, hold = split(series, HOLDOUT)

    out = pd.DataFrame({"timestamp": hold.index.values})
    for t in TARGETS:
        y = train[t].values
        name = pick_model(y, HOLDOUT)
        print(f"{t}: selected {name}")
        fc = CANDIDATES[name](y, HOLDOUT)
        out[t] = fc if np.isfinite(fc).all() else CANDIDATES["last_value"](y, HOLDOUT)

    out["active_vm_count"] = out["active_vm_count"].clip(lower=0)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0, 100)
    validate_predictions(out, hold.index.values)
    out.to_csv(a.output, index=False)


if __name__ == "__main__":
    main()