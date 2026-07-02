# Capacity Aging V3 Diagnostic Report

## Purpose

This report diagnoses the cycle-level capacity-aging extension and the corrected V3 forecast. The objective is a reproducible deterministic surrogate forecast for SOH, not a Bayesian posterior, qmax/R0 electrochemical model, uncertainty model, or visual curve drawing exercise.

The main successful project result remains the voltage-response surrogate workflow from examples 11-16: a physics-inspired voltage-response baseline, residual learning, temperature ablation, and multi-battery evaluation. The temperature ablation result is battery-dependent; temperature helps in some cases but is not consistently beneficial. Example 17 is an exploratory capacity-aging extension after that main workflow.

## Root Cause Found

The previous V3 collapsed toward straight-line post-split forecasts because:

- The self-only forecast was dominated by a split-point degradation rate term.
- Train-validation often selected weak acceleration, so the curvature term was too small.
- The conservative slope rule suppressed curvature to avoid B0005/B0007 over-drop.
- Fleet blending was either zero or blended full prior levels/slopes, so it either did nothing or risked overriding target-specific evidence.

The revised V3 changes this:

- Self-only uses a smooth nonlinear degradation-rate transition after the split.
- The forecast preserves value continuity and first-derivative continuity at the split.
- The post-split forecast is not `y_split + m * dx`; slope changes across the forecast horizon.
- Fleet contributes only a curvature residual after the split, not a level or split-slope override.
- Nonlinear candidates are selected using only target train-internal validation; target test labels are not used.

## Data Pipeline Checks

### Source Files

- Input sequence data: `results/discharge_sequences_all.csv`
- Input sequence summary: `results/discharge_sequences_all_summary.csv`
- Current V3 outputs:
  - `results/capacity_aging_v3_predictions.csv`
  - `results/capacity_aging_v3_metrics.csv`
  - `results/capacity_aging_v3_components.csv`

### Capacity and SOH

- `examples/11_build_discharge_sequence_dataset.py` reads each NASA MATLAB battery file and stores discharge time-step sequences.
- MATLAB discharge-cycle `Capacity` is copied to each time-step row in the cycle.
- V3 aggregates by `battery_id` and `discharge_index`.
- Cycle capacity is the robust median valid positive `capacity_ah` within a discharge cycle.
- Initial capacity is the median of the first 3 valid positive cycle capacities.
- `SOH = capacity_ah / initial_capacity_ah`.

### Energy Age

- `cycle_energy_kwh` is summed from voltage, absolute current, and time step duration.
- `energy_age_kwh` is the cumulative sum of cycle energy.
- `energy_frac = energy_age_kwh / max energy_age_kwh` per battery.

Important caveat: `energy_frac` uses the full-life energy maximum of each battery. This is acceptable for retrospective normalized-age comparison, but a strict prospective deployment would need absolute energy age or a known future usage horizon.

### Target Data Summary

| battery | cycles | train | test | initial Ah | final Ah | final SOH | split energy frac | raw SOH increases | max raw recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0005 | 168 | 117 | 51 | 1.84633 | 1.32508 | 0.71768 | 0.74358 | 36 | 0.04784 |
| B0006 | 168 | 117 | 51 | 2.02514 | 1.18568 | 0.58548 | 0.75533 | 27 | 0.07501 |
| B0007 | 168 | 117 | 51 | 1.88066 | 1.43246 | 0.76168 | 0.73274 | 47 | 0.05220 |
| B0018 | 132 | 92 | 40 | 1.84320 | 1.34105 | 0.72757 | 0.73244 | 21 | 0.07120 |

Raw SOH recovery spikes are not smoothed or deleted. Monotonicity applies only to prediction trends.

## Leakage Check

V3 uses only:

- target first 70% observed SOH/capacity;
- target train-internal validation tail;
- target energy/cycle covariates;
- fleet curves from non-target batteries.

V3 does not use:

- target test SOH labels;
- target final SOH;
- visual correction from the held-out region;
- test data for calibration, smoothing, hyperparameter selection, or blend selection.

## Fleet-Prior Diagnostics

Fleet curves are useful but not uniformly reliable:

| target | top fleet batteries | weighted-prior train bias | weighted-prior test bias | target recent slope | prior recent slope | diagnosis |
|---|---|---:|---:|---:|---:|---|
| B0005 | B0006, B0018, B0007, B0036, B0032 | -0.02301 | +0.00807 | -0.47907 | -0.30975 | Prior is shallower than target late trend. |
| B0006 | B0018, B0007, B0005, B0036, B0032 | -0.04060 | +0.01988 | -0.43230 | -0.25578 | Prior underfits training level and is too shallow. |
| B0007 | B0018, B0006, B0005, B0036, B0055 | +0.00093 | +0.00102 | -0.35659 | -0.28202 | Prior is close but slightly shallower. |
| B0018 | B0007, B0006, B0005, B0036, B0032 | -0.00220 | -0.01737 | -0.07922 | -0.34439 | Prior is much steeper than target recent trend. |

Because of this, the corrected V3 does not blend the full weighted-prior level curve into the final model. Fleet contributes only curvature residuals after the split, preserving the target-specific split value and first derivative.

## Current V3 Forecast Diagnostics

### Test Metrics

| battery | self RMSE | fleet RMSE | self MAE | fleet MAE | fleet blend | final bias fleet |
|---|---:|---:|---:|---:|---:|---:|
| B0005 | 0.03295 | 0.03295 | 0.02666 | 0.02666 | 0.00 | -0.08487 |
| B0006 | 0.00904 | 0.00904 | 0.00729 | 0.00729 | 0.00 | -0.01831 |
| B0007 | 0.02260 | 0.01762 | 0.01857 | 0.01414 | 0.30 | -0.04936 |
| B0018 | 0.01602 | 0.01602 | 0.01322 | 0.01322 | 0.00 | -0.00541 |

Average test RMSE:

- `self_only_shape_constrained`: 0.0202 SOH
- `fleet_updated_deterministic_v3`: 0.0189 SOH
- relative improvement: approximately 6.4%

Fleet-updated V3 gives a modest average improvement, not a strong or decisive improvement. The benefit is battery-dependent. It helps B0007 in this split, is effectively neutral for B0005/B0006/B0018 under current validation selection, and should be described as an exploratory deterministic extension rather than a robust final capacity-aging solution.

### Straight-Line Collapse Diagnostics

| battery | model | split slope | train-tail slope | forecast avg slope | final slope | slope change | train-tail curvature | forecast curvature | effectively linear |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0005 | self/fleet | -0.4446 | -0.4799 | -0.5114 | -0.6898 | -0.3276 | 0.002295 | 0.000032 | False |
| B0006 | self/fleet | -0.3809 | -0.4339 | -0.4300 | -0.6084 | -0.3250 | 0.001745 | 0.000027 | False |
| B0007 | self | -0.3592 | -0.3573 | -0.3797 | -0.5023 | -0.2434 | 0.001898 | 0.000024 | False |
| B0007 | fleet | -0.3657 | -0.3573 | -0.3424 | -0.4497 | -0.3067 | 0.001898 | 0.000226 | False |
| B0018 | self/fleet | -0.1898 | -0.1343 | -0.1956 | -0.2331 | -0.0688 | 0.001665 | 0.000012 | False |

No current V3 target triggers the near-linear forecast warning.

## V1 vs V2 vs Corrected V3

| version | model | B0005 | B0006 | B0007 | B0018 | mean RMSE |
|---|---|---:|---:|---:|---:|---:|
| V1 | fleet_prior_calibrated | 0.01634 | 0.01311 | 0.00602 | 0.03465 | 0.01753 |
| V2 | fleet_updated_deterministic | 0.02009 | 0.01707 | 0.01628 | 0.01847 | 0.01798 |
| V3 corrected | fleet_updated_deterministic_v3 | 0.03295 | 0.00904 | 0.01762 | 0.01602 | 0.01891 |
| V3 corrected self | self_only_shape_constrained | 0.03295 | 0.00904 | 0.02260 | 0.01602 | 0.02015 |

Interpretation:

- V1 has strong RMSE but poorer visual interpretability and more calibration-heavy behavior.
- V2 is clean and deterministic but often close to self-only and can still look line-like.
- Corrected V3 is the most conceptually aligned with the requested nonlinear deterministic forecast, but the gain over self-only is modest. B0006 and B0007 are relatively reasonable, while B0005 and B0018 still show important limitations.

## Recommendation

Use corrected V3 as the final capacity-aging extension only with conservative wording:

- final evaluated fleet candidate: `fleet_updated_deterministic_v3`;
- baseline: `self_only_shape_constrained`;
- conclusion: deterministic nonlinear aging forecast is feasible, and fleet curvature gives a modest average improvement, but the benefit is battery-dependent.

Do not claim a robust or decisive fleet improvement. State that the capacity-aging extension is interpretable and useful as a secondary result, but it is not robust enough to be the main success of the project.

## Remaining Limitations

- B0005 remains underpredicted in the late-life tail.
- B0006 is relatively reasonable in the current split.
- B0007 is relatively reasonable and benefits from fleet curvature, but still has some final underprediction.
- B0018 still misses local post-split recovery / plateau behavior; the monotone deterministic trend cannot represent those local recoveries.
- The deterministic fleet-informed model is not robust enough to be the main success of the project.
- The normalized energy axis is retrospective unless future cumulative energy horizon is known.

## Regenerated Files

- `results/capacity_aging_v3_predictions.csv`
- `results/capacity_aging_v3_metrics.csv`
- `results/capacity_aging_v3_components.csv`
- `figures/capacity_aging_v3_model_comparison.png`
- `figures/capacity_aging_v3_fleet_diagnostics.png`
- `figures/capacity_aging_v3_rmse_summary.png`

## Validation

Commands run:

```bash
PATH=.venv/bin:$PATH python examples/17_v3_capacity_aging_forecast_models.py
PATH=.venv/bin:$PATH python -m py_compile examples/17_v3_capacity_aging_forecast_models.py
```

Current `git status --short`:

```text
?? capacity_aging_v3_diagnostic_report.md
?? examples/17_v2_fleet_update_capacity_aging.py
?? examples/17_v3_capacity_aging_forecast_models.py
```
