import numpy as np
from cwfc.targets import *
#from cwfc.baselines import CANDIDATES
from cwfc.candidates import CANDIDATES
from cwfc.rolling import *
#from cwfc.validation import *

series = build_targets(load_readings("challenge_data"))
train, hold = split(series)          # don't touch `hold` again until the very end
y = train["aggregate_cpu_utilization"].values
base = summarize(rolling_origin_eval(y, CANDIDATES["last_value"])[0])
for name, f in CANDIDATES.items():
    s = summarize(rolling_origin_eval(y, f)[0])
    print(f"{name:15s} MAE={s['mae']:.4f} RMSE={s['rmse']:.4f} vs last_value MAE: {1 - s['mae']/base['mae']:+.1%}")

# per-fold comparison and error by horizon step
errs_em, _ = rolling_origin_eval(y, CANDIDATES["expanding_mean"])
errs_lv, _ = rolling_origin_eval(y, CANDIDATES["last_value"])
errs_ar, _ = rolling_origin_eval(y, CANDIDATES["ar1_meanrev"])

print("per-fold MAE  expanding_mean:", np.abs(errs_em).mean(axis=1).round(3))
print("per-fold MAE  last_value    :", np.abs(errs_lv).mean(axis=1).round(3))
print("folds where expanding_mean wins:", (np.abs(errs_em).mean(1) < np.abs(errs_lv).mean(1)).sum(), "of", len(errs_em))

for name, e in [("last_value", errs_lv), ("expanding_mean", errs_em), ("ar1_meanrev", errs_ar)]:
    print(name, "MAE by horizon step (1,6,12,24):", np.abs(e).mean(axis=0)[[0, 5, 11, 23]].round(3))