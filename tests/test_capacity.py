import numpy as np
import pandas as pd
import pytest

import hbpinn_battery.capacity as capacity
from hbpinn_battery.capacity import (
    MonotonePchipCurve,
    build_cycle_dataset,
    enforce_forecast_sanity,
    evaluate_target,
    fleet_residual_calibrated_prediction,
    score_fleet_curves,
    select_final_update_family,
    select_self_hyperparameters,
    split_at_normalized_energy,
)


def synthetic_sequences(battery_count=4, cycle_count=30):
    rows = []
    for battery_index in range(battery_count):
        battery_id = f"B{battery_index:04d}"
        for cycle in range(cycle_count):
            capacity_ah = 2.0 - 0.012 * cycle - 0.002 * battery_index * cycle
            for step, temperature in enumerate((20.0 + cycle, 24.0 + cycle)):
                rows.append(
                    {
                        "battery_id": battery_id,
                        "discharge_index": cycle,
                        "capacity_ah": capacity_ah,
                        "cycle_energy_kwh": 0.01 + 0.0001 * battery_index,
                        "temperature_c": temperature,
                        "step": step,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cycle_dataset():
    dataset, skipped = build_cycle_dataset(synthetic_sequences())
    assert skipped == {}
    return dataset


def test_build_cycle_dataset_aggregation_and_ages(cycle_dataset):
    battery = cycle_dataset[cycle_dataset["battery_id"] == "B0000"]
    first = battery.iloc[0]
    assert len(battery) == 30
    assert first["number_of_steps"] == 2
    assert first["mean_temperature_c"] == 22.0
    assert first["min_temperature_c"] == 20.0
    assert first["max_temperature_c"] == 24.0
    assert np.allclose(battery["energy_age_kwh"], np.arange(1, 31) * 0.01)
    assert battery["energy_frac"].iloc[-1] == pytest.approx(1.0)


def test_initial_capacity_uses_first_three_positive_capacity_median():
    table = synthetic_sequences(battery_count=1)
    capacities = [3.0, 2.0, 4.0]
    for cycle, value in enumerate(capacities):
        table.loc[table["discharge_index"] == cycle, "capacity_ah"] = value
    dataset, _ = build_cycle_dataset(table)
    assert dataset["initial_capacity_ah"].iloc[0] == 3.0
    assert dataset["SOH"].iloc[1] == pytest.approx(2.0 / 3.0)


def test_split_at_normalized_energy_boundary():
    table = pd.DataFrame({"energy_frac": [0.1] * 12 + [0.70] + [0.71] * 6})
    train, test = split_at_normalized_energy(table)
    assert (train["energy_frac"] <= 0.70).all()
    assert (test["energy_frac"] > 0.70).all()
    assert 0.70 in train["energy_frac"].to_numpy()


def test_monotone_curve_never_increases():
    curve = MonotonePchipCurve([0.0, 0.5, 1.0], [1.0, 1.1, 0.8])
    prediction = curve(np.linspace(0.0, 1.0, 21))
    assert np.all(np.diff(prediction) <= 0.0)


def test_enforce_forecast_sanity_is_finite_and_monotone():
    prediction = enforce_forecast_sanity(
        [1.0, 0.98, 0.99, 0.95, 0.96], 3, "synthetic", "B0000"
    )
    assert np.isfinite(prediction).all()
    assert np.all(np.diff(prediction) <= 0.0)


def test_score_fleet_curves_components_and_normalized_weights(cycle_dataset):
    train, _ = split_at_normalized_energy(
        cycle_dataset[cycle_dataset["battery_id"] == "B0000"]
    )
    curves = capacity.build_fleet_curves(cycle_dataset, "B0000")
    components = score_fleet_curves(curves, train)
    expected = {
        "similarity_score",
        "overall_train_rmse",
        "late_train_rmse",
        "slope_mismatch",
        "anchor_shift",
        "selected_top_k",
        "weight",
    }
    assert expected.issubset(components.columns)
    assert components.loc[components["selected_top_k"], "weight"].sum() == pytest.approx(1.0)


def test_residual_calibration_uses_target_minus_prior(monkeypatch, cycle_dataset):
    train, _ = split_at_normalized_energy(
        cycle_dataset[cycle_dataset["battery_id"] == "B0000"]
    )
    curves = capacity.build_fleet_curves(cycle_dataset, "B0000")
    eval_x = train["energy_frac"].to_numpy()
    captured = {}
    original = capacity.residual_update_curve

    def capture(train_x, residual_train, evaluation_x):
        captured["residual"] = residual_train.copy()
        return original(train_x, residual_train, evaluation_x)

    monkeypatch.setattr(capacity, "residual_update_curve", capture)
    _, prior, _ = fleet_residual_calibrated_prediction(train, eval_x, curves)
    expected = train["SOH"].to_numpy() - prior[: len(train)]
    assert np.allclose(captured["residual"], expected)


def test_validation_gate_contains_fallback_and_exact_blend_weights(cycle_dataset):
    train, _ = split_at_normalized_energy(
        cycle_dataset[cycle_dataset["battery_id"] == "B0000"]
    )
    params, _ = select_self_hyperparameters(train)
    curves = capacity.build_fleet_curves(cycle_dataset, "B0000")
    selected = select_final_update_family(train, "B0000", params, curves)
    summary = pd.DataFrame(selected["validation_summary"])
    assert "self_only_fallback" in set(summary["update_family"])
    blends = summary.loc[
        summary["update_family"] == "fleet_curvature_residual", "blend_weight"
    ].tolist()
    assert blends == [0.10, 0.20, 0.30]


def test_evaluate_target_runs_in_memory(cycle_dataset):
    result = evaluate_target(cycle_dataset, "B0000")
    assert result["battery_id"] == "B0000"
    assert len(result["self_pred"]) == 30
    assert len(result["fleet_pred"]) == 30
    assert set(result["metrics"]["model_name"]) == {
        "self_only_baseline",
        "fleet_updated_deterministic_final",
    }
