import argparse
import numpy as np
import pandas as pd
from .targets import load_readings, build_targets, split, HOLDOUT, TARGETS
#from .baselines import CANDIDATES
#from .validation import rolling_origin_eval, summarize
from .candidates import CANDIDATES
from .rolling import rolling_origin_eval, summarize

def pick_model(y, horizon):
    """Select by rolling-origin RMSE on the training window only. Ties go to the first candidate."""
    min_train = max(2, min(48, len(y) - horizon - 1))
    best, best_score = None, np.inf
    for name, f in CANDIDATES.items():
        errs, _ = rolling_origin_eval(y, f, horizon, min_train)
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
        out[t] = CANDIDATES[name](y, HOLDOUT)
    out["active_vm_count"] = out["active_vm_count"].clip(lower=0)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0, 100)
    assert len(out) == HOLDOUT and np.isfinite(out.drop(columns="timestamp").values).all()
    out.to_csv(a.output, index=False)

if __name__ == "__main__":
    main()