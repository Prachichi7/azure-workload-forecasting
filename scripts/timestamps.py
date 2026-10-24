import pandas as pd, numpy as np
r = pd.read_csv("challenge_data/vm_cpu_readings.csv.gz")
m = pd.read_csv("challenge_data/vm_metadata.csv.gz")

# range sanity
print(r[["min_cpu", "avg_cpu", "max_cpu"]].describe())
print(((r.min_cpu > r.avg_cpu) | (r.avg_cpu > r.max_cpu)).sum())  # ordering violations

# the aggregate target
agg = r.groupby("timestamp").avg_cpu.mean()
print(agg.describe())
print(agg.iloc[:5], agg.iloc[-5:])
print(agg.idxmax(), agg.max(), agg.idxmin(), agg.min())

# train vs holdout level
print(agg.iloc[:110].mean(), agg.iloc[110:].mean())
print(agg.iloc[110:].index[[0, -1]])   # expect 33000 and 39900

# metadata
print(m.isna().sum())
print(m.vm_category.value_counts())
print(m.virtual_core_bucket.value_counts())
print(m.memory_gb_bucket.value_counts())
print(m[["timestamp_vm_created", "timestamp_vm_deleted"]].describe())



import matplotlib.pyplot as plt
import numpy as np
ts = np.sort(r.timestamp.unique())
agg = r.groupby("timestamp").avg_cpu.mean()
tr = agg.loc[ts[:110]]                      # training window only

print(tr.describe())
print(tr.idxmax(), tr.max(), tr.idxmin(), tr.min())
print(tr.autocorr(1), tr.autocorr(2), tr.autocorr(6))
print(tr.diff().std(), tr.std())
tr.plot(); 
plt.savefig("reports/figures/agg_train.png")   # mkdir reports\figures first

import os
os.makedirs("reports/figures", exist_ok=True)
tr.plot()
plt.savefig("reports/figures/agg_train.png", dpi=150, bbox_inches="tight")
plt.close()
print(tr.describe())
print(tr.idxmax(), tr.max(), tr.idxmin(), tr.min())
print(tr.autocorr(1), tr.autocorr(2), tr.autocorr(6))
print(tr.diff().std(), tr.std())