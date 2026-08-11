import numpy as np
import pandas as pd

import hbpinn_battery.voltage as voltage
from hbpinn_battery.voltage import (
    BASELINE_FEATURES,
    RESIDUAL_FEATURES,
    TEMPERATURE_RESIDUAL_FEATURES,
    HybridVoltageModel,
    build_voltage_features,
    fit_hybrid_voltage_model,
    predict_hybrid_voltage,
    split_discharge_cycles,
)


def synthetic_voltage_table(number_of_cycles=6):
    rows = []
    for cycle in range(number_of_cycles):
        for step, used_capacity in enumerate([0.0, 0.6, 2.4]):
            capacity = 2.0
            current = [-2.0, -1.5, 0.2][step]
            soc = np.clip(1.0 - used_capacity / capacity, 0.0, 1.0)
            discharge_current = max(-current, 0.0)
            baseline = 3.0 + 0.8 * soc - 0.15 * soc**2 + 0.03 * soc**3
            baseline -= 0.04 * discharge_current
            residual = 0.002 * cycle + 0.001 * step
            rows.append(
                {
                    "discharge_index": cycle,
                    "step_index": step,
                    "capacity_ah": capacity,
                    "used_capacity_ah": used_capacity,
                    "current_a": current,
                    "voltage_v": baseline + residual,
                    "cumulative_energy_kwh": 0.1 * cycle + 0.01 * step,
                    "temperature_c": 22.0 + cycle + 0.5 * step,
                }
            )
    return pd.DataFrame(rows)


def test_soc_calculation_and_clipping():
    table = synthetic_voltage_table(number_of_cycles=1)
    baseline, _ = build_voltage_features(table)

    expected = (1.0 - table["used_capacity_ah"] / table["capacity_ah"]).clip(
        lower=0.0,
        upper=1.0,
    )
    np.testing.assert_allclose(baseline["soc"], expected)
    np.testing.assert_allclose(baseline["soc"], [1.0, 0.7, 0.0])


def test_feature_column_order_without_temperature():
    baseline, residual = build_voltage_features(synthetic_voltage_table())

    assert baseline.columns.tolist() == BASELINE_FEATURES
    assert residual.columns.tolist() == RESIDUAL_FEATURES
    assert "temperature_c" not in residual.columns


def test_feature_column_order_with_temperature():
    _, residual = build_voltage_features(
        synthetic_voltage_table(),
        include_temperature=True,
    )

    assert residual.columns.tolist() == TEMPERATURE_RESIDUAL_FEATURES
    assert residual.columns[-1] == "temperature_c"


def test_split_discharge_cycles_is_chronological_and_keeps_cycles_together():
    table = synthetic_voltage_table(number_of_cycles=10).sample(
        frac=1.0,
        random_state=5,
    )
    train, test = split_discharge_cycles(table)

    assert sorted(train["discharge_index"].unique()) == list(range(7))
    assert sorted(test["discharge_index"].unique()) == list(range(7, 10))
    assert set(train["discharge_index"]).isdisjoint(test["discharge_index"])
    assert len(train) == 7 * 3
    assert len(test) == 3 * 3


def test_training_residual_sign_is_measured_minus_baseline(monkeypatch):
    table = synthetic_voltage_table(number_of_cycles=2)
    captured = {}

    class Baseline:
        def predict(self, X):
            return np.full(len(X), 3.5)

    class Residual:
        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(voltage, "fit_voltage_baseline", lambda X, y: Baseline())

    def capture_residual(X, residual):
        captured["residual"] = np.asarray(residual)
        return Residual()

    monkeypatch.setattr(voltage, "fit_rf_residual_model", capture_residual)
    fit_hybrid_voltage_model(table)

    np.testing.assert_allclose(
        captured["residual"],
        table["voltage_v"].to_numpy() - 3.5,
    )


def test_hybrid_prediction_is_baseline_plus_residual():
    table = synthetic_voltage_table(number_of_cycles=1)

    class Baseline:
        def predict(self, X):
            return np.arange(len(X), dtype=float) + 3.0

    class Residual:
        def predict(self, X):
            return np.arange(len(X), dtype=float) * 0.1

    model = HybridVoltageModel(
        baseline_model=Baseline(),
        residual_model=Residual(),
        include_temperature=False,
        baseline_feature_names=list(BASELINE_FEATURES),
        residual_feature_names=list(RESIDUAL_FEATURES),
    )
    predictions = predict_hybrid_voltage(model, table)

    np.testing.assert_allclose(
        predictions["hybrid_prediction"],
        predictions["baseline_prediction"] + predictions["residual_prediction"],
    )


def test_fit_and_predict_selected_rf_hybrid():
    table = synthetic_voltage_table()
    model = fit_hybrid_voltage_model(table, include_temperature=False)
    predictions = predict_hybrid_voltage(model, table)

    assert model.include_temperature is False
    assert model.residual_feature_names == RESIDUAL_FEATURES
    assert all(len(values) == len(table) for values in predictions.values())
    assert all(np.isfinite(values).all() for values in predictions.values())


def test_fit_and_predict_temperature_aware_rf_hybrid():
    table = synthetic_voltage_table()
    model = fit_hybrid_voltage_model(table, include_temperature=True)
    predictions = predict_hybrid_voltage(model, table)

    assert model.include_temperature is True
    assert model.residual_feature_names == TEMPERATURE_RESIDUAL_FEATURES
    assert all(len(values) == len(table) for values in predictions.values())
    assert all(np.isfinite(values).all() for values in predictions.values())
