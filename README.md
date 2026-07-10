# HB-PINN Battery

Final SCE project for Li-ion battery prognosis using the NASA battery dataset.

The project has two completed parts:

1. Voltage-response prediction for discharge cycles using physics-informed and hybrid residual models.
2. Cycle-level capacity-aging prognosis using fleet-informed deterministic updates.

The implementation is packaged as reusable Python code, with reproducible example scripts that generate the metrics, predictions, plots, and final summaries used for the project submission.

## Package Structure

```text
.
├── src/hbpinn_battery/   # reusable package code
├── examples/            # reproducible scripts from data inspection to final summaries
├── results/             # generated CSV metrics and predictions
├── figures/             # generated plots
├── outputs/             # generated plots and final summary outputs
└── tests/               # minimal import/install test
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest -q
```

## Data

Place the NASA battery dataset under:

```text
nasa_raw_data/
```

Raw NASA data files are expected locally and are not part of the package logic. The package and examples read from this local data directory, while generated metrics and plots are written to `results/`, `figures/`, and `outputs/`.

## Reproduction

The final project results can be regenerated with:

```bash
python examples/16_fit_multi_battery_hybrid_models.py
python examples/17_capacity_aging_final.py
python examples/18_full_partial_observation_experiments.py
python examples/19_final_summary.py
```

Earlier numbered scripts in `examples/` provide the supporting data inspection, extraction, baseline modeling, and plotting workflow.

## Final Results

Voltage-response prediction:

- The best B0005 voltage result comes from the temperature ablation using the hybrid residual model without temperature: about `0.0194 V` mean test RMSE.
- The temperature-aware hybrid voltage model is close, but temperature does not consistently improve test performance: about `0.0204 V` on the B0005 ablation and about `0.0335 V` mean test RMSE in the multi-battery summary.
- In the multi-battery voltage summary, the non-temperature hybrid residual model averages about `0.0325 V` test RMSE, compared with about `0.0788 V` for the physics-inspired baseline.

Capacity-aging prognosis:

- The self-only capacity-aging baseline has about `0.0239 SOH` mean test RMSE.
- The final fleet-informed deterministic capacity-aging model improves this to about `0.0200 SOH`.
- The observation update experiment gives a modest average improvement with partial sparse updates, about `0.0225 SOH` mean RMSE, but the effect is battery-dependent. B0018 remains difficult and degrades under the tested update schemes.

## Limitations

- The NASA dataset is small, so model comparisons are sensitive to battery selection and train/test splits.
- Aging behavior is battery-dependent; fleet information helps some cells but is not universally accepted by the deterministic update rule.
- Temperature features do not consistently improve voltage prediction.
- Capacity recovery and non-monotonic local behavior make long-horizon aging prediction difficult.
- Observation updates are deterministic diagnostics, not a full Bayesian posterior update.

