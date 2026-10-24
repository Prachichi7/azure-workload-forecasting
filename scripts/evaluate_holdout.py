"""FINAL TEST SCORE. Run once, after the model and policy are frozen.
Nothing here feeds back into model selection or tuning."""
import numpy as np
import pandas as pd

from cwfc.analysis import mase, peak_mae, rel_improvement, residual_band
from cwfc.candidates import CANDIDATES
from cwfc.metrics import mae, rmse, smape
from cwfc.predict import pick_model
from cwfc.rolling import rolling_origin_eval
from cwfc.targets import HOLDOUT as H, TARGETS, build_targets, load_readings, split

series = build_targets(load_readings("challenge_data"))
train, hold = split(series)
rows = []

for t in TARGETS:
    y, truth = train[t].values, hold[t].values
    name = pick_model(y, H)                                  # selected on the training window only
    fc, base = CANDIDATES[name](y, H), CANDIDATES["last_value"](y, H)
    row = {"target": t, "model": name, "MAE": mae(truth, fc), "RMSE": rmse(truth, fc),
           "mean_signed_error": float(np.mean(truth - fc))}
    if t == "aggregate_cpu_utilization":
        errs, acts = rolling_origin_eval(y, CANDIDATES[name], H, 48, 6)
        errs_b, _ = rolling_origin_eval(y, CANDIDATES["last_value"], H, 48, 6)
        p_m, p_b = acts - errs, acts - errs_b
        lo, hi = residual_band(errs)
        m_model, _ = residual_band(errs, lo=0.9, hi=0.9)
        m_max, _ = residual_band(acts - np.maximum(p_m, p_b), lo=0.9, hi=0.9)
        row.update({
            "sMAPE_%": smape(truth, fc), "MASE": mase(truth, fc, y),
            "baseline_MAE": mae(truth, base), "baseline_RMSE": rmse(truth, base),
            "MAE_improvement_vs_last_value": rel_improvement(truth, fc, base),
            "peak_MAE_top10pct": peak_mae(truth, fc),
            "band80_coverage": float(((truth >= fc + lo) & (truth <= fc + hi)).mean()),
            "miss_rate_model_q90": float((truth > fc + m_model).mean()),
            "miss_rate_max_rule_q90": float((truth > np.maximum(fc, base) + m_max).mean()),
        })
    rows.append(row)

out = pd.DataFrame(rows)
print(out.round(4).T.to_string())
out.to_csv("reports/holdout_results.csv", index=False)