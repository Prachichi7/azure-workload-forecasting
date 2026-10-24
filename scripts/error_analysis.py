from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cwfc.analysis import mase, peak_mae, rel_improvement, residual_band
from cwfc.candidates import CANDIDATES
from cwfc.metrics import mae, rmse, smape            # starter functions
from cwfc.rolling import rolling_origin_eval, rolling_origins
from cwfc.targets import build_targets, load_readings, split

BASE = "last_value"
H, MIN_TRAIN, STEP = 24, 48, 6
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)

series = build_targets(load_readings("challenge_data"))
train, _ = split(series)                             # validation only; holdout untouched
y = train["aggregate_cpu_utilization"].values

from cwfc.predict import pick_model
MODEL = pick_model(y, H, MIN_TRAIN, STEP)           # same rule as the submission pipeline
print("analysed model:", MODEL)

e_m, a = rolling_origin_eval(y, CANDIDATES[MODEL], H, MIN_TRAIN, STEP)
e_b, _ = rolling_origin_eval(y, CANDIDATES[BASE], H, MIN_TRAIN, STEP)
p_m, p_b = a - e_m, a - e_b                          # errs = actual - forecast

rows = []
for name, p in [(MODEL, p_m), (BASE, p_b)]:
    rows.append({"model": name, "MAE": mae(a, p), "RMSE": rmse(a, p), "sMAPE_%": smape(a, p),
                 "MASE": mase(a, p, y), "peak_MAE_q90": peak_mae(a, p)})
rows[0]["MAE_improvement_vs_base"] = rel_improvement(a, p_m, p_b)
table = pd.DataFrame(rows)
print(table.round(4).to_string(index=False))
table.to_csv("reports/error_analysis.csv", index=False)

# error by horizon step
steps = np.arange(1, H + 1)
plt.plot(steps, np.abs(e_m).mean(0), label=MODEL)
plt.plot(steps, np.abs(e_b).mean(0), label=BASE)
plt.xlabel("horizon step (5 min)"); plt.ylabel("MAE (pp)"); plt.legend()
plt.savefig(FIG / "mae_by_horizon.png", dpi=150, bbox_inches="tight"); plt.close()

# series with fold boundaries, plus per-fold error (failure analysis)
origins = rolling_origins(len(y), H, MIN_TRAIN, STEP)
print("per-fold MAE:", {int(o): round(float(v), 3) for o, v in zip(origins, np.abs(e_m).mean(1))})

plt.plot(y)
for o in origins:
    plt.axvline(o, alpha=0.2)
plt.xlabel("index in training window"); plt.ylabel("mean CPU (%)")
plt.savefig(FIG / "train_series_folds.png", dpi=150, bbox_inches="tight"); plt.close()

# 80% band and in-sample coverage (optimistic: same folds that estimated it)
lo, hi = residual_band(e_m)
print("in-sample 80% band coverage:", ((a >= p_m + lo) & (a <= p_m + hi)).mean().round(3))

# --- peak behaviour: signed error (positive = under-forecast) ---
mask = a >= np.quantile(a, 0.9)
print("\npeak points:", int(mask.sum()), "of", a.size)
print("mean signed error at peaks  model:", round(float((a - p_m)[mask].mean()), 3),
      " base:", round(float((a - p_b)[mask].mean()), 3))
print("share of points under-forecast  model:", round(float((a > p_m).mean()), 3),
      " base:", round(float((a > p_b).mean()), 3))

# --- capacity-rule trade-off on validation: cap = forecast + margin ---
level = float(y.mean())
print(f"\nlevel {level:.2f}%  | cap = forecast + margin (margin = pooled error quantile)")

# --- one pre-specified variant: cap on the higher of model and persistence forecasts ---
p_hi = np.maximum(p_m, p_b)
e_hi = a - p_hi
peak = a >= np.quantile(a, 0.9)
rows2 = []
for label, p, e in [("model+margin", p_m, e_m), ("max(model,persistence)+margin", p_hi, e_hi)]:
    for q in (0.8, 0.9, 0.95):
        m, _ = residual_band(e, lo=q, hi=q)
        cap = p + m
        rows2.append({"rule": label, "quantile": q, "margin_pp": float(m.mean()),
                      "miss_rate": float((a > cap).mean()),
                      "peak_miss_rate": float((a[peak] > cap[peak]).mean()),
                      "mean_shortfall_pp": float(np.maximum(a - cap, 0).mean()),
                      "mean_spare_pp": float(np.maximum(cap - a, 0).mean())})
variant = pd.DataFrame(rows2)
print("\n", variant.round(3).to_string(index=False))
variant.to_csv("reports/policy_variants.csv", index=False)

rows = []
for q in (0.5, 0.8, 0.9, 0.95):
    m, _ = residual_band(e_m, lo=q, hi=q)
    cap = p_m + m
    rows.append({"quantile": q, "margin_pp": float(m.mean()),
                 "margin_pct_of_level": float(m.mean() / level * 100),
                 "miss_rate": float((a > cap).mean()),
                 "mean_shortfall_pp": float(np.maximum(a - cap, 0).mean()),
                 "mean_spare_pp": float(np.maximum(cap - a, 0).mean())})
policy = pd.DataFrame(rows)
print(policy.round(3).to_string(index=False))
policy.to_csv("reports/policy_tradeoff.csv", index=False)