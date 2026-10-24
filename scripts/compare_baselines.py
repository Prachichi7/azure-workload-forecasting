import numpy as np

from cwfc.candidates import CANDIDATES
from cwfc.rolling import rolling_origin_eval, summarize
from cwfc.targets import build_targets, load_readings, split

series = build_targets(load_readings("challenge_data"))
train, _ = split(series)                     # the holdout is never touched in this script
y = train["aggregate_cpu_utilization"].values

res = {n: rolling_origin_eval(y, f)[0] for n, f in CANDIDATES.items()}
lv = np.abs(res["last_value"])
print(f"{len(lv)} folds, horizon 24\n")
for n, e in sorted(res.items(), key=lambda kv: summarize(kv[1])["rmse"]):
    s = summarize(e)
    fold = np.abs(e).mean(axis=1)
    wins = int((fold < lv.mean(axis=1)).sum())
    h = np.abs(e).mean(axis=0)[[0, 5, 11, 23]]
    print(f"{n:15s} MAE={s['mae']:.4f} RMSE={s['rmse']:.4f} "
          f"vs_last={1 - s['mae'] / lv.mean():+.1%} fold_wins={wins}/{len(fold)} "
          f"MAE@h1,6,12,24={np.round(h, 3)}")