import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]

def validate_predictions(pred: pd.DataFrame, expected_timestamps) -> None:
    if list(pred.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}, got {list(pred.columns)}")
    if len(pred) != len(expected_timestamps):
        raise ValueError(f"expected {len(expected_timestamps)} rows, got {len(pred)}")
    if not np.array_equal(pred["timestamp"].to_numpy(), np.asarray(expected_timestamps)):
        raise ValueError("timestamps must equal the holdout timestamps, in chronological order")
    if pred["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps")
    if not np.isfinite(pred[COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite predictions")
    if (pred["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    if not pred["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be in [0, 100]")