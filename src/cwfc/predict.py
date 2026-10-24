from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .candidates import CANDIDATES
from .rolling import rolling_origin_eval, summarize
from .submission import validate_predictions
from .targets import HOLDOUT, TARGETS, build_targets, load_readings, split


def pick_model(y, horizon, min_train=48, step=6):
    """Select by rolling-origin RMSE on the training window only.
    Ties go to the first candidate (last_value). Too little history -> last_value."""
    best, best_score = "last_value", np.inf
    for name, f in CANDIDATES.items():
        errs, _ = rolling_origin_eval(y, f, horizon, min_train, step)
        if errs.size == 0:
            return "last_value"
        score = summarize(errs)["rmse"]
        if score < best_score - 1e-12:
            best, best_score = name, score
    return best


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