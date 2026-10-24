# Cloud Workload Forecasting: Azure Public Dataset V2 (curated subset)

End-to-end forecasting pipeline for the Atomity AI/ML take-home. It builds two fleet-level
time series from 10,000 Azure VM traces, validates forecasting methods with leakage-safe
rolling-origin evaluation, forecasts the final 2-hour holdout, and turns the forecast into
a capacity-planning recommendation.

- Targets: `active_vm_count` and `aggregate_cpu_utilization` (percent, in [0, 100])
- Cadence: native 5-minute grid, 134 timestamps (110 training, 24 holdout)
- Final report: `reports/final_report.md`
- Required output: `predictions.csv` (24 rows)

## 1. Requirements and installation

Python 3.11 or newer is required.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Runtime dependencies (declared in `pyproject.toml`): numpy, pandas, scikit-learn,
statsmodels, matplotlib. `pytest` is installed through the `dev` extra. An optional
`notebooks` extra installs JupyterLab and is not needed to run the pipeline.

## 2. Dataset

The dataset is the fixed `data-v1` release of the challenge repository. It is not
committed to this repository.

1. Download `data-v1.tar.gz` from
   https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1
   (direct link:
   https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz).
2. Optionally verify the archive. Its SHA-256 is
   `d04d402ac1719c1f5b869e3699fc3d57f6f8273751f1ef19bedb29397b39caeb`.
   - Windows: `Get-FileHash data-v1.tar.gz -Algorithm SHA256`
   - Linux / macOS: `sha256sum data-v1.tar.gz`
3. Extract it into `challenge_data/` at the repository root:

```bash
mkdir challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

Expected layout:

```text
challenge_data/
  vm_cpu_readings.csv.gz
  vm_metadata.csv.gz
  DATA_CARD.md
  SHA256SUMS
```

The prediction pipeline reads only `vm_cpu_readings.csv.gz` (columns `timestamp`, `vm_id`,
`avg_cpu`). The metadata file is used only by the exploratory analysis.

## 3. Generate the predictions (one command)

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This builds both target series, holds out the final 24 timestamps, selects a model for each
target using rolling-origin validation on the first 110 timestamps only, forecasts the 24
holdout timestamps, validates the file against the submission contract, and writes
`predictions.csv` with the columns `timestamp,active_vm_count,aggregate_cpu_utilization`
(no index column). The selected model for each target is printed to the console.

The output is validated before it is written: exactly 24 rows, timestamps equal to the
holdout timestamps in chronological order, no duplicates, finite values,
`active_vm_count >= 0`, and `aggregate_cpu_utilization` in [0, 100].

The pipeline does not depend on the size of the supplied data. It derives the holdout from
the last 24 timestamps of whatever readings file it is given, so a small smoke dataset with
the same columns works as long as it has more than 26 timestamps.

## 4. Reproduce the analysis

| Purpose | Command |
|---|---|
| Data understanding and workload figures | `python -m cwfc.eda --data challenge_data` |
| Compare all candidate models (rolling origin) | `python scripts/compare_baselines.py` |
| Error analysis, peak behavior, capacity-policy trade-off | `python scripts/error_analysis.py` |
| Score the frozen model on the holdout (run once) | `python scripts/evaluate_holdout.py` |
| Test suite | `python -m pytest -q` |

Figures are written to `reports/figures/` and result tables to `reports/`. Run all commands
from the repository root.

## 5. Main modeling decisions

**Targets.** `active_vm_count` is the number of distinct `vm_id` values with a valid
`avg_cpu` reading at each timestamp. `aggregate_cpu_utilization` is the mean of `avg_cpu`
across those VMs, clipped to [0, 100]. The full 134-point series is built first, then split
chronologically (first 110 / last 24).

**Missing data.** Rows with a missing `avg_cpu` are dropped before aggregation. Nothing is
imputed. This release has no missing readings, no duplicate (`vm_id`, `timestamp`) pairs,
and a perfect 300-second grid.

**Cadence.** Modeling is done directly at the required 5-minute cadence, so no mapping
from a coarser cadence back to five minutes is needed.

**`active_vm_count` is constant by construction.** All 10,000 VMs are present at all 134
timestamps. Every VM has a creation timestamp of 0 and none is deleted inside the window,
because the release was built from complete VM traces. The pipeline still treats the target
generically: all candidates are scored, and ties go to persistence (last value).

**Validation.** Rolling-origin evaluation with horizon 24 on the 110-point training window:
origins 50, 56, ..., 86 (seven folds), where the last fold ends exactly at index 110. The
holdout is never part of any fold. All models, including the baselines, go through the same
evaluation code.

**Candidates.** Baselines: last value, moving averages (6, 12, 24), expanding mean, EWMA,
drift, seasonal-naive (12, 24). Models: AR(1) with mean reversion, per-horizon direct AR(1),
ARIMA(1,0,1), and per-horizon ridge regression on lag features. Every model is a function
of the history it is given and estimates all parameters from that history only.

**Model selection.** Lowest rolling-origin RMSE on the training window. Among candidates
within 2 percent of the best RMSE, the earliest (simplest) candidate in the list wins. This
guards against selecting a model on noise, since seven overlapping folds cannot separate
close candidates.

**Leakage controls.**
- The target series is split before any model is fit, and only the first 110 points are
  passed to model fitting and selection.
- Features use only data available at the forecast origin (lags and rolling statistics
  looking backward).
- Whole-trace metadata columns (`avg_cpu`, `max_cpu`, `p95_max_cpu`,
  `timestamp_vm_deleted`) summarize 30 days and include the holdout period, so they are
  not used as features or for segmentation. Only the static columns `vm_category`,
  `virtual_core_bucket` and `memory_gb_bucket` are used.
- A test overwrites the holdout readings with extreme values and asserts that the
  predictions do not change.

**Uncertainty and capacity policy.** Prediction margins come from pooled quantiles of
rolling-origin forecast errors (horizon steps pooled in buckets of six, because seven folds
per step is too few). The capacity-rule trade-off between under- and over-provisioning is
analysed in `scripts/error_analysis.py`, and the recommendation is in the final report.

## 6. Validation results (training window, rolling origin, horizon 24)

`aggregate_cpu_utilization`, errors in percentage points:

| Model | MAE | RMSE |
|---|---|---|
| last value (baseline) | 0.2152 | 0.2715 |
| expanding mean (baseline) | 0.2088 | 0.2447 |
| AR(1) mean reversion (selected) | 0.2009 | 0.2373 |
| direct AR(1) | 0.1967 | 0.2339 |
| ARIMA(1,0,1) | 0.1973 | 0.2368 |
| ridge on lags | 0.2220 | 0.2581 |

The three best models are within noise of each other, so the simplest of them is selected.
Mean-reverting models beat persistence by about 7 percent in MAE, but not in every fold.
Persistence is best for the first few steps and degrades with horizon, while mean-based
models are flat. All models under-forecast at the highest-load points. See
`reports/final_report.md` for the full analysis and the holdout results.

## 7. Repository layout

```text
README.md
pyproject.toml
predictions.csv                 Required prediction file
src/cwfc/
  baselines.py                  Starter baselines (unchanged)
  validation.py                 Starter validation utilities (unchanged)
  metrics.py                    Starter metrics: MAE, RMSE, sMAPE, normalized MAE (unchanged)
  targets.py                    Target construction and chronological split
  candidates.py                 Candidate forecasting functions
  rolling.py                    Rolling-origin evaluation
  analysis.py                   MASE, relative improvement, peak error, residual bands
  submission.py                 Submission-contract validator
  predict.py                    Entry point: python -m cwfc.predict
  eda.py                        Data understanding and workload figures
scripts/
  compare_baselines.py          Candidate comparison table
  error_analysis.py             Error analysis and capacity-policy trade-off
  evaluate_holdout.py           One-time holdout scoring of the frozen model
tests/                          Starter tests plus pipeline, leakage and validator tests
reports/
  final_report.md               Final report
  figures/                      Generated figures
```

The starter modules `baselines.py`, `validation.py` and `metrics.py` are left unchanged;
the new modules are added alongside them, and `metrics.py` is reused by the analysis code.

## 8. Tests

```bash
python -m pytest -q
```

The tests cover target construction (duplicates counted once, missing values dropped), the
110/24 chronological split, rolling-origin boundaries, the submission validator, model
selection on a constant series, an end-to-end run on a small synthetic dataset, and the
holdout-independence check described above.

## 9. Assumptions and limitations

- The data covers about 11 hours, so daily or weekly seasonality cannot be learned, and
  seasonal-naive baselines are included only as a reference.
- Selection among 14 candidates on seven heavily overlapping folds is noisy, which is why a
  tolerance rule is used. Validation scores of the selected model are optimistic; the
  holdout score, computed once on the frozen model, is the honest estimate.
- Empirical margins and bands are estimated on the same folds they are evaluated on, so
  their validation coverage is optimistic.
- The release contains only VMs with complete traces, so there is no VM churn in the window.
  Results do not transfer directly to a fleet with real creation and deletion activity.
- Early exploratory analysis printed some statistics over the full 134-point series. All
  modeling, selection and policy decisions were made from the first 110 points only.

## 10. Data source and license

Microsoft Azure Public Dataset V2: https://github.com/Azure/AzurePublicDataset.
The dataset is not redistributed in this repository. Code is released under the MIT license
of the challenge repository.
