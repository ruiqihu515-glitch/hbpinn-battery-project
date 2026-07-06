# Capacity-Aging Final Example 17

## 1. Objective

This final Example 17 consolidates the earlier V1/V2/V3 capacity-aging experiments into one reproducible cycle-level aging workflow. The voltage-response model remains the main successful result of the project. Capacity aging is an exploratory extension with modest, battery-dependent improvement.

## 2. Data pipeline

The script reads `results/discharge_sequences_all.csv`, aggregates time-step discharge records to one row per battery discharge cycle, filters invalid capacity records, and evaluates B0005, B0006, B0007, and B0018. It uses the existing cleaned discharge-sequence data and does not use the Example 18 observation-update outputs.

## 3. SOH and energy-age definitions

SOH is defined as `capacity_ah / initial_capacity_ah`, where `initial_capacity_ah` is the median of the first valid positive capacities when at least three are available. This avoids making the whole SOH scale depend on one potentially noisy first point.

`cycle_energy_kwh` is the cycle energy already computed from voltage, absolute current, and time-step duration in the discharge-sequence dataset. Cumulative energy age is the cumulative sum of cycle energy. Normalized energy age is `cumulative_energy_age / full-life max cumulative energy` per battery.

The normalized full-life energy age is a retrospective normalized-age comparison. Strict prospective deployment would need absolute energy age, a known future usage horizon, or another deployable age coordinate that does not depend on the target battery's future full-life maximum.

## 4. Train/test split

The target battery is split at normalized energy age x = 0.70. Only rows with x <= 0.70 are used for fitting, calibration, similarity scoring, and train-internal validation. Test rows with x > 0.70 are used only for evaluation and plotting.

## 5. Self-only baseline model

The baseline is a target-only shape-constrained forecast. It fits a monotone smooth PCHIP/isotonic trend on target training SOH, estimates recent degradation rate from the training tail, and extrapolates with a low-complexity nonlinear degradation-rate continuation. Hyperparameters are selected by train-internal validation only.

## 6. Fleet-informed deterministic update model

The final model uses non-target battery curves as fleet diagnostics and limited future-shape guidance. Individual fleet curves are aligned to the target training anchor, scored by training-shape similarity, and combined into uniform and similarity-weighted diagnostic priors. Train-only rolling validation chooses between a self-only fallback, a fleet-curvature residual update, and a fleet-prior residual calibration. The final forecast is deterministic and reproducible, and fleet information is accepted only when the train-only validation folds support it.

It does not fully reproduce the Bayesian posterior/vMLP method in the reference paper. It does not model qmax/R0 Bayesian posterior or partial-observation Bayesian update.

## 7. Why V1/V2/V3 were consolidated

V1 had the clearest method diagnostic figure because it showed individual fleet curves, uniform prior, weighted prior, and calibrated target forecast. V2/V3 cleaned up the forecast comparison and leakage controls. This final script keeps the V1-style diagnostic figure but uses the cleaner V3-style monotone nonlinear forecast and train-internal model selection.

## 8. Final selected result

The final fleet-informed model improves over the self-only baseline on average test RMSE: self-only = 0.02394, final = 0.01997, relative change = 16.57%. The gain is modest and battery-dependent.

| battery | self RMSE | final RMSE | improvement | note |
| --- | ---: | ---: | ---: | --- |
| B0005 | 0.02949 | 0.02949 | 0.00% | self_only_fallback: fleet update was not accepted or did not improve versus self-only; late-life final SOH bias remains material; tail behavior/local recovery remains difficult |
| B0006 | 0.01248 | 0.01248 | 0.00% | self_only_fallback: fleet update was not accepted or did not improve versus self-only; first 70% target fit is noisy or imperfect |
| B0007 | 0.02477 | 0.00891 | 64.02% | fleet_prior_residual_calibration: late-life final SOH bias remains material |
| B0018 | 0.02900 | 0.02900 | 0.00% | self_only_fallback: fleet update was not accepted or did not improve versus self-only; late-life final SOH bias remains material; first 70% target fit is noisy or imperfect; tail behavior/local recovery remains difficult |

## 9. Per-battery findings

B0005 and B0018 remain difficult late-life targets. Their tails and local recovery/plateau behavior are not fully represented by the monotone low-complexity forecast. B0006 can show imperfect first-70% fit while still producing a comparatively good future forecast; this should be interpreted as battery-dependent behavior, not a general success claim. B0007 is the clearest case where fleet curvature can help under this split.

## 10. Limitations

The fleet-informed model gives only modest and battery-dependent improvement. It is not a full Bayesian HBPINN reproduction, not a vMLP posterior method, and not a qmax/R0 posterior update. It is still useful because it completes the project pipeline from voltage response to cycle-level aging evaluation in a reproducible way, with explicit train/test separation and clear diagnostic figures.

Skipped non-target batteries during cleaning:

- B0033: median SOH above reasonable range; max SOH above reasonable range; min SOH below reasonable range
- B0038: median SOH above reasonable range; max SOH above reasonable range
- B0039: median SOH above reasonable range; max SOH above reasonable range
- B0040: median SOH above reasonable range; max SOH above reasonable range
- B0041: max SOH above reasonable range
- B0042: missing essential numeric fields; min SOH below reasonable range
- B0043: missing essential numeric fields; min SOH below reasonable range
- B0044: missing essential numeric fields; min SOH below reasonable range
- B0045: missing essential numeric fields
- B0046: missing essential numeric fields
- B0047: missing essential numeric fields
- B0048: missing essential numeric fields
- B0049: missing essential numeric fields; max SOH above reasonable range
- B0050: missing essential numeric fields; max SOH above reasonable range; min SOH below reasonable range
- B0051: missing essential numeric fields; max SOH above reasonable range
- B0052: too few valid cycles; missing essential numeric fields
- B0053: missing essential numeric fields
- B0054: missing essential numeric fields
