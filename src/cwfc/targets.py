from pathlib import Path
import pandas as pd

HOLDOUT = 24
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]

def load_readings(data_dir):
    return pd.read_csv(Path(data_dir) / "vm_cpu_readings.csv.gz")

def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp. Rows with missing avg_cpu are dropped (not imputed);
    a VM is counted once per timestamp via nunique."""
    df = readings.dropna(subset=["avg_cpu"])
    g = df.groupby("timestamp")
    out = pd.DataFrame({
        "active_vm_count": g["vm_id"].nunique().astype(float),
        "aggregate_cpu_utilization": g["avg_cpu"].mean().clip(0, 100),
    })
    return out.sort_index()

def split(series, holdout=HOLDOUT):
    """Chronological split. The tail `holdout` rows are never used for fitting or selection."""
    return series.iloc[:-holdout], series.iloc[-holdout:]