from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from .targets import HOLDOUT, load_readings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    r = load_readings(a.data)
    ts = np.sort(r["timestamp"].unique())
    gaps = np.unique(np.diff(ts), return_counts=True)
    notes = [f"rows={len(r)}  vms={r['vm_id'].nunique()}  timestamps={len(ts)}",
             f"gaps (seconds: count)={dict(zip(map(int, gaps[0]), map(int, gaps[1])))}",
             f"duplicates={int(r.duplicated(['vm_id', 'timestamp']).sum())}  "
             f"missing avg_cpu={int(r['avg_cpu'].isna().sum())}",
             f"VMs per timestamp: min={int(r.groupby('timestamp')['vm_id'].nunique().min())}"]

    r = r[r["timestamp"].isin(ts[:-HOLDOUT])]            # EDA on the training window only
    g = r.groupby("timestamp")["avg_cpu"]
    agg = g.mean()

    # 1. aggregate series
    agg.plot(); plt.ylabel("mean avg_cpu (%)")
    plt.savefig(out / "eda_aggregate.png", dpi=150, bbox_inches="tight"); plt.close()

    # 2. skewed per-VM distribution (log y)
    r.groupby("vm_id")["avg_cpu"].mean().hist(bins=60); plt.yscale("log")
    plt.xlabel("per-VM mean CPU (%)")
    plt.savefig(out / "eda_vm_cpu_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

    # 3. autocorrelation of the aggregate
    plot_acf(agg.values, lags=24)
    plt.savefig(out / "eda_acf.png", dpi=150, bbox_inches="tight"); plt.close()

    # 4. tail series: the mean hides what capacity teams care about
    fig, ax = plt.subplots(2, 1, sharex=True)
    g.quantile(0.95).plot(ax=ax[0], ylabel="p95 CPU (%)")
    g.apply(lambda s: (s > 80).mean() * 100).plot(ax=ax[1], ylabel="% VMs > 80%")
    fig.savefig(out / "eda_tail.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 5. segments + 6. lifetimes (need metadata; static columns only for segmentation)
    mp = Path(a.data) / "vm_metadata.csv.gz"
    if mp.exists():
        m = pd.read_csv(mp)
        for col in ("vm_category", "virtual_core_bucket"):
            seg = r.merge(m[["vm_id", col]], on="vm_id").groupby(["timestamp", col])["avg_cpu"].mean().unstack()
            seg.plot(); plt.ylabel("mean avg_cpu (%)")
            plt.savefig(out / f"eda_segment_{col}.png", dpi=150, bbox_inches="tight"); plt.close()
            notes.append(f"{col}: mean CPU by segment = {seg.mean().round(2).to_dict()}")
        notes.append(f"created after window start: {int((m['timestamp_vm_created'] > ts[0]).sum())}; "
                     f"deleted within window: {int((m['timestamp_vm_deleted'] <= ts[-1]).sum())}")

    Path("reports/eda_summary.md").write_text("\n".join(f"- {n}" for n in notes), encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()