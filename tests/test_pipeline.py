import numpy as np
import pandas as pd
import pytest

from cwfc import predict
from cwfc.candidates import CANDIDATES
from cwfc.rolling import rolling_origins
from cwfc.submission import COLUMNS, validate_predictions
from cwfc.targets import HOLDOUT, build_targets, split


def _readings(n_vm=30, n_ts=134, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "timestamp": np.tile(np.arange(n_ts) * 300, n_vm),
        "vm_id": np.repeat([f"vm{i}" for i in range(n_vm)], n_ts),
    })
    df["avg_cpu"] = np.clip(rng.gamma(2, 4, len(df)), 0, 100)
    df["min_cpu"] = df["avg_cpu"] * 0.5
    df["max_cpu"] = np.clip(df["avg_cpu"] * 2, 0, 100)
    return df


def test_targets_count_vm_once_and_drop_nan():
    df = pd.DataFrame({"timestamp": [0, 0, 0, 300], "vm_id": ["a", "a", "b", "a"],
                       "avg_cpu": [10.0, 10.0, np.nan, 20.0]})
    t = build_targets(df)
    assert t["active_vm_count"].tolist() == [1, 1]
    assert t["aggregate_cpu_utilization"].tolist() == [10.0, 20.0]


def test_split_is_chronological_110_24():
    tr, ho = split(build_targets(_readings()))
    assert len(tr) == 110 and len(ho) == HOLDOUT
    assert tr.index.max() < ho.index.min()


def test_rolling_origins_stay_inside_training_window():
    o = rolling_origins(110, 24, 48, 6)
    assert o[-1] + 24 == 110 and o[0] >= 48
    assert rolling_origins(20, 24, 48, 6) == []


def test_candidates_use_only_the_history_they_are_given():
    y = np.random.default_rng(1).normal(9, 0.2, 110)
    y_future_changed = y.copy(); y_future_changed[80:] = 999
    for name, f in CANDIDATES.items():
        assert np.allclose(f(y[:80].copy(), 24), f(y_future_changed[:80].copy(), 24)), name


def test_validator_accepts_good_and_rejects_bad():
    ts = np.arange(24) * 300 + 33000
    good = pd.DataFrame({"timestamp": ts, "active_vm_count": 10000.0,
                         "aggregate_cpu_utilization": 9.4})
    validate_predictions(good, ts)
    for bad in (good.assign(aggregate_cpu_utilization=101.0), good.assign(active_vm_count=-1.0),
                good.iloc[:-1], good.rename(columns={"timestamp": "t"})):
        with pytest.raises(ValueError):
            validate_predictions(bad, ts)


def test_end_to_end_on_small_synthetic_dataset(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    _readings().to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    out = tmp_path / "pred.csv"
    monkeypatch.setattr("sys.argv", ["predict", "--data", str(data), "--output", str(out)])
    predict.main()
    p = pd.read_csv(out)
    assert list(p.columns) == COLUMNS and len(p) == HOLDOUT

def test_pick_model_constant_series_is_last_value():
    assert predict.pick_model(np.full(110, 10000.0), 24) == "last_value"