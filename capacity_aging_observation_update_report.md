# Example 18: Full / Partial Observation Experiments

Example 18 completes the planned full / partial observation experiment as an exploratory continuation of final Example 17. It starts from the final Example 17 capacity-aging forecast for the same target batteries, then asks whether additional target-battery SOH observations can correct the future long-horizon forecast. It uses the same cleaned SOH/capacity pipeline, normalized cumulative-energy age definition, and 70% train/test split as final Example 17.

The base forecast is frozen after the Example 17 fit from the first 70% of target normalized energy age. The `no_update` case continues that final Example 17 forecast without using any target observations after x = 0.70. The `partial_sparse_update` case observes only 4 sparse target points in 0.70 < x <= 0.80. The `full_window_update` case observes all target SOH points in 0.70 < x <= 0.80. All metrics are computed only on the future evaluation region x > 0.80.

This is not a full Bayesian posterior update. There is no posterior distribution, sampling, uncertainty propagation, qmax/R0 latent-state inference, or likelihood update. The update is deterministic and reproducible: it fits a small ridge-regularized residual correction, offset plus local slope, between the observed update SOH and the frozen no-update forecast.

Data leakage is avoided by construction. For each target battery, the initial forecast is the final Example 17 forecast trained from target history at x <= 0.70. The update cases use target observations only from 0.70 < x <= 0.80, and evaluation excludes the history and update windows. No SOH labels after x = 0.80 are used for fitting, tuning, smoothing, shifting, or calibration.

Mean RMSE by case:

| case | mean RMSE (SOH) |
| --- | ---: |
| no_update | 0.02349 |
| partial_sparse_update | 0.02252 |
| full_window_update | 0.02288 |

Per-battery improvement versus no update:

| battery | no update RMSE | partial RMSE | full RMSE | partial improvement | full improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0005 | 0.03537 | 0.03226 | 0.02815 | 8.79% | 20.40% |
| B0006 | 0.01405 | 0.01130 | 0.01462 | 19.60% | -4.05% |
| B0007 | 0.00936 | 0.00809 | 0.00672 | 13.55% | 28.16% |
| B0018 | 0.03518 | 0.03843 | 0.04201 | -9.26% | -19.42% |

The best empirical mean result is the sparse partial update: mean RMSE improves from about 0.0235 for `no_update` to about 0.0225 for `partial_sparse_update`. The full-window update is also slightly better than no update on average, at about 0.0229, but it is not better than the sparse update. The improvement is modest and battery-dependent, so this should not be presented as a decisive success.

B0018 is the main limitation: both observation updates worsen the future forecast there, showing that deterministic updates remain sensitive to late-life battery-specific behavior and local recovery/non-stationarity. The full-window update is not necessarily better than sparse observation because fitting more local noisy behavior in the update window can hurt future extrapolation after x = 0.80.

For the final report discussion, Example 18 should be described as an exploratory but logically complete full / partial observation experiment extending final Example 17. Sparse partial observations can improve aging forecasts on average, but deterministic updates remain sensitive to late-life battery-specific behavior and should not be framed as a decisive capacity-aging success.
