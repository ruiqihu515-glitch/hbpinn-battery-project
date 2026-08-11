# Hybrid Battery Prognosis

## Overview

This research-software project studies lithium-ion battery voltage response and
capacity aging using the NASA battery dataset. The `hbpinn_battery` package
exposes the verified experimental workflow as reusable Python functions and
classes for:

- raw discharge-cycle preprocessing;
- physics-inspired voltage modeling;
- Random-Forest (RF) voltage-residual learning, with an optional temperature
  feature;
- cycle-level capacity and state-of-health (SOH) forecasting;
- self-only and fleet-informed capacity candidates with train-only validation;
- deterministic post-forecast correction from later target observations.

The final reusable implementation combines physics-inspired baseline modeling
with deterministic data-driven residual and aging corrections.

## Key Results

### Voltage Response

The final Example-16 evaluation covers 22 batteries with chronological
cycle-level train/test splits.

| Model | Mean Test RMSE [V] |
| --- | ---: |
| Physics-inspired baseline | 0.07878 |
| RF hybrid without temperature | **0.03247** |
| Temperature-aware RF hybrid | 0.03350 |

The RF hybrid without temperature has the lowest mean RMSE among these final
RF variants. The temperature-aware model remains available as a reusable,
evaluated feature variant. The MLP residual remains an evaluated diagnostic in
Example 16 but is not part of the package API or selected final model.

### Why did temperature not improve the aggregate RF result?

Adding temperature did not provide a consistent improvement across batteries.
The temperature-aware RF improved 11 of the 22 evaluated batteries, worsened 9,
and left 2 nearly unchanged; the slightly higher aggregate RMSE was strongly
influenced by a large degradation for B0041. Temperature is also highly
correlated with existing state variables such as SOC and used capacity, so much
of its information is already represented by the non-temperature feature set.
For the four batteries used in the final capacity study, temperature worsened
B0005 and B0007, was nearly neutral for B0006, and slightly improved B0018. This
indicates that the temperature effect is battery-dependent and provides limited
robust incremental information for the fixed RF formulation. Therefore, the
non-temperature RF was retained as the final voltage model.

### Capacity Aging

Capacity forecasting is evaluated after a split at normalized cumulative
energy age 0.70.

| Battery | Self RMSE [SOH] | Final RMSE [SOH] | Selected family |
| --- | ---: | ---: | --- |
| B0005 | 0.02949 | 0.02949 | `self_only_fallback` |
| B0006 | 0.01248 | 0.01248 | `self_only_fallback` |
| B0007 | 0.02477 | **0.00891** | `fleet_prior_residual_calibration` |
| B0018 | 0.02900 | 0.02900 | `self_only_fallback` |
| **Mean** | **0.02394** | **0.01997** | — |

Only B0007 selects a fleet-prior residual-calibration update. Fleet information
is not automatically applied to every target.

### Observation Update

Example 18 evaluates optional deterministic corrections to the frozen final
capacity forecast.

| Strategy | Four-target Mean RMSE [SOH] |
| --- | ---: |
| No update | 0.02349 |
| Partial sparse update | **0.02252** |
| Full-window update | 0.02288 |

The sparse update gives the best four-target mean in this experiment, but the
improvement is not universal: both update strategies worsen B0018.

## Package Structure

```text
src/hbpinn_battery/
├── __init__.py
├── data.py
├── preprocessing.py
├── metrics.py
├── voltage.py
├── capacity.py
└── observation.py
```

- `data.py`: MATLAB file discovery, loading, and conversion utilities.
- `preprocessing.py`: conversion of raw NASA discharge cycles into aligned
  time-step sequence and extraction-summary tables.
- `metrics.py`: shared RMSE and MAE implementations.
- `voltage.py`: voltage features, chronological cycle splitting, the
  physics-inspired baseline, RF residual learning, the temperature-aware RF
  option, and hybrid fitting/prediction.
- `capacity.py`: cycle-level aging data, self forecasts, fleet curves,
  scoring and priors, fleet residual calibration, and validation-gated final
  selection.
- `observation.py`: deterministic observation updates applied to a frozen
  capacity forecast.

## Installation

Normal editable installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and testing, install the optional `dev` dependencies:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The distribution name is `hbpinn-battery`; the Python import package is
`hbpinn_battery`. Python 3.10 or newer is required.

## Data

Raw NASA `.mat` files are not bundled with this repository. Installing the
package and running the default test suite do not require raw NASA data.
Existing processed and result artifacts support review and reproduction of
the reported results.

When NASA `.mat` files are supplied separately, `preprocessing.py` can rebuild
the discharge-sequence dataset. The verified processed dataset contains 34
batteries, 637,352 time-step rows, and 16 columns. External data paths are
provided by the caller; the package does not require a hard-coded raw-data
directory.

## Quick Start

### Preprocessing

```python
from hbpinn_battery.preprocessing import (
    extract_discharge_sequences,
    trim_discharge_to_min_voltage,
)

# mat_files is an iterable of user-supplied NASA .mat file paths.
sequence_table, summary_table = extract_discharge_sequences(mat_files)
```

`trim_discharge_to_min_voltage` retains the first minimum-voltage point and
aligns the four arrays to their shortest remaining length. It preserves the
Example-11 behavior: it does not apply Example-06 current filtering and does
not re-zero time.

### Voltage Modeling

```python
from hbpinn_battery.voltage import (
    build_voltage_features,
    split_discharge_cycles,
    fit_voltage_baseline,
    fit_rf_residual_model,
    fit_hybrid_voltage_model,
    predict_hybrid_voltage,
)

train_table, test_table = split_discharge_cycles(battery_sequence_table)

model = fit_hybrid_voltage_model(
    train_table,
    include_temperature=False,
)
temperature_model = fit_hybrid_voltage_model(
    train_table,
    include_temperature=True,
)

prediction = predict_hybrid_voltage(model, test_table)
temperature_prediction = predict_hybrid_voltage(temperature_model, test_table)
```

`build_voltage_features` constructs the feature matrices;
`split_discharge_cycles` performs the chronological cycle split;
`fit_voltage_baseline` fits the physics-inspired linear component;
`fit_rf_residual_model` fits an RF residual correction;
`fit_hybrid_voltage_model` is the baseline-plus-RF training convenience
function; and `predict_hybrid_voltage` applies an already fitted hybrid model.

### Capacity Forecasting

```python
from hbpinn_battery.capacity import (
    build_cycle_dataset,
    nonlinear_self_forecast,
    build_fleet_curves,
    score_fleet_curves,
    fleet_prior_prediction,
    fleet_residual_calibrated_prediction,
    select_final_update_family,
    evaluate_target,
)

cycle_dataset, skipped = build_cycle_dataset(sequence_table)
result = evaluate_target(cycle_dataset, "B0007")
```

`evaluate_target` coordinates the energy-based split, self fitting, fleet
construction and scoring, train-only family selection, predictions, metrics,
and diagnostics. The final forecast may remain self-only.

### Observation Update

```python
from hbpinn_battery.observation import (
    split_by_energy_fraction,
    choose_sparse_update_points,
    fit_residual_correction,
    apply_residual_update,
    evaluate_observation_update,
)

observation_result = evaluate_observation_update(
    cycle_dataset,
    "B0005",
)
```

### Metrics

```python
from hbpinn_battery.metrics import mae, rmse
```

## Model Details

### Voltage Model

The clipped state-of-charge feature is

```math
\mathrm{SOC} = 1-\frac{Q_{\mathrm{used}}}{Q_{\mathrm{capacity}}}, \qquad \mathrm{SOC}\in[0,1].
```

The physics-inspired baseline is

```math
V_{\mathrm{base}} = \beta_0 + \beta_1\mathrm{SOC} + \beta_2\mathrm{SOC}^2 + \beta_3\mathrm{SOC}^3 + \beta_4 I.
```

where `discharge_current_a` supplies the non-negative discharge-current
feature. The residual and hybrid prediction are

```math
r = V_{\mathrm{measured}} - V_{\mathrm{base}}.
```

```math
\hat V_{\mathrm{hybrid}} = V_{\mathrm{base}} + \hat r_{\mathrm{RF}}.
```

The exact ordered feature sets are:

- baseline: `soc`, `soc_squared`, `soc_cubed`,
  `discharge_current_a`;
- RF residual without temperature: the baseline fields followed by
  `used_capacity_ah`, `cumulative_energy_kwh`, `discharge_index`;
- temperature-aware RF: the same RF residual fields followed by
  `temperature_c`.

The RF uses `n_estimators=200`, `max_depth=12`, `random_state=42`, and
`n_jobs=-1`. `include_temperature=False` selects the normal/final RF hybrid
variant; `include_temperature=True` selects the evaluated temperature-aware
variant.

### Capacity Forecasting and Validation Gate

Cycle-level SOH is defined by

```math
\mathrm{SOH} = \frac{Q_{\mathrm{cycle}}}{Q_{\mathrm{initial}}}.
```

The aging coordinate is normalized cumulative energy:

```math
x_E = \frac{E_{\mathrm{cumulative}}}{E_{\mathrm{cumulative,max}}}.
```

The denominator is the observed full-life cumulative-energy maximum, making
this coordinate retrospective rather than fully prospective. The base
train/test threshold is `x_E = 0.70`.

The validation gate compares these candidate families using training data
only:

- `self_only_fallback`, which is always available;
- `fleet_curvature_residual`, with curvature blend strengths 0.10, 0.20,
  and 0.30;
- `fleet_prior_residual_calibration`.

Fleet candidates must satisfy the implemented validation eligibility criteria.
The final test set is not used for family selection, and fleet information is
not forced. The values 0.10/0.20/0.30 are curvature blend strengths, not
observation-time fractions.

Temperature mean/minimum/maximum statistics are retained in the cycle-level
dataset for analysis, but temperature is not a capacity-prediction feature in
the final capacity workflow.

### Observation-Based Post-Forecast Update

The observation workflow is a separate deterministic post-forecast
experiment. It operates on the frozen final capacity prediction and does not
retrain the capacity model. It is neither part of the base capacity model nor
a Bayesian update.

| Region | Definition |
| --- | --- |
| History | `energy_frac <= 0.70` |
| Observation/update | `0.70 < energy_frac <= 0.80` |
| Future evaluation | `energy_frac > 0.80` |

The sparse strategy selects up to four observations using the Example-18
selection rule. The full-window strategy uses every observation in the update
window. Both fit a deterministic residual correction to the frozen forecast;
metrics are calculated only in the future evaluation region.

The shared metrics are

```math
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}.
```

```math
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|.
```

## Reproducing the Experiments

The authoritative final workflow is:

```text
NASA .mat files
  → Example 11 / preprocessing.py
  → processed discharge-sequence data

processed data
  → Example 16 / voltage.py
  → final multi-battery voltage evaluation

processed data
  → final Example 17 / capacity.py
  → final capacity-aging forecasts
  → Example 18 / observation.py (optional post-forecast update)

final result tables
  → Example 19
  → final summaries
```

The authoritative scripts are:

```bash
python examples/11_build_discharge_sequence_dataset.py
python examples/16_fit_multi_battery_hybrid_models.py
python examples/17_capacity_aging_final.py
python examples/18_full_partial_observation_experiments.py
python examples/19_final_summary.py
```

These scripts are the authoritative experiment entry points. Raw-data-dependent
scripts require the NASA `.mat` files to be supplied separately and available
at the input location expected by the corresponding script. Examples 01–10
document staged data inspection and early baselines, while standalone Examples
12–15 retain the B0005 diagnostic/ablation lineage. The authoritative final
workflow is Examples 11, 16, `17_capacity_aging_final.py`, 18, and 19. Obsolete
intermediate Example-17 implementations are not included in the final
submission.

The reusable package was checked against these authoritative scripts. The
refactor preserved their preprocessing, algorithms, splits, selected models,
predictions, tables, and reported numerical results.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The verified suite contains 46 passing tests. Tests use synthetic data and do
not require NASA `.mat` files. They cover preprocessing, metrics, voltage,
capacity, observation updates, and package imports.

## Outputs and Results

Core model metrics, predictions, reports, and final figures are tracked for
inspection after cloning. The large processed sequence table,
`results/discharge_sequences_all.csv`, and its extraction summary,
`results/discharge_sequences_all_summary.csv`, are generated locally from
separately supplied NASA `.mat` files and are intentionally not tracked. The
Example 19 final summary tables are included as lightweight review artifacts.

Important final artifacts include:

- `results/multi_battery_voltage_response_metrics.csv` and
  `results/multi_battery_voltage_response_per_cycle_metrics.csv`;
- `results/capacity_aging_final_metrics.csv` and
  `results/capacity_aging_final_predictions.csv`;
- `results/capacity_aging_observation_update_metrics.csv` and
  `results/capacity_aging_observation_update_predictions.csv`;
- `outputs/example_19_final_summary/final_voltage_summary.csv`,
  `outputs/example_19_final_summary/final_capacity_summary.csv`, and
  `outputs/example_19_final_summary/final_overview_summary.csv`;
- `figures/multi_battery_voltage_rmse_summary.png`,
  `figures/capacity_aging_final_forecast_comparison.png`,
  `figures/capacity_aging_final_rmse_summary.png`, and
  `figures/capacity_aging_observation_update_rmse_summary.png`.

Final tracked metrics, predictions, diagnostic comparisons, and figures are
included for review. Obsolete intermediate capacity-model artifacts are
excluded from the final submission.

## Limitations

- Normalized energy age uses the observed full-life cumulative-energy maximum
  and is therefore retrospective.
- Monotonicity constraints apply to fitted trends and predictions; raw
  measured SOH observations are not rewritten.
- Fleet information is validation-gated and is not selected for every target.
- Observation updates do not improve every battery; B0018 worsens under both
  sparse and full-window updates in the final experiment.

## Repository Structure

```text
hbpinn-battery-project/
├── src/hbpinn_battery/   # reusable package
├── tests/                # synthetic package tests
├── examples/             # experiment and reproduction scripts
├── results/              # numerical result artifacts
├── figures/              # research figures
├── outputs/              # final summary tables and figures
├── pyproject.toml
└── README.md
```
