import numpy as np
import pandas as pd

import hbpinn_battery.observation as observation
from hbpinn_battery.observation import (
    apply_residual_update,
    choose_sparse_update_points,
    evaluate_observation_update,
    fit_residual_correction,
    split_by_energy_fraction,
)


def synthetic_cycles():
    x = np.round(np.arange(0.0, 1.001, 0.025), 3)
    return pd.DataFrame(
        {
            "battery_id": "BTEST",
            "discharge_index": np.arange(len(x)),
            "energy_frac": x,
            "SOH": 1.0 - 0.2 * x,
        }
    )


def test_split_and_observation_window_boundaries(monkeypatch):
    table = synthetic_cycles()
    base = 1.0 - 0.18 * table["energy_frac"].to_numpy()
    monkeypatch.setattr(observation, "evaluate_target", lambda *_: {"fleet_pred": base})

    result = evaluate_observation_update(table, "BTEST")

    assert result["history"]["energy_frac"].max() == 0.70
    assert result["future"]["energy_frac"].min() > 0.70
    assert result["full_window_update"]["energy_frac"].min() > 0.70
    assert result["full_window_update"]["energy_frac"].max() == 0.80


def test_evaluation_region_starts_strictly_after_cutoff(monkeypatch):
    table = synthetic_cycles()
    base = 1.0 - 0.18 * table["energy_frac"].to_numpy()
    monkeypatch.setattr(observation, "evaluate_target", lambda *_: {"fleet_pred": base})

    result = evaluate_observation_update(table, "BTEST")
    evaluation = result["predictions"].query("region == 'evaluation'")

    assert evaluation["normalized_energy_age"].min() > 0.80
    assert result["metrics"]["n_eval_points"].eq(len(evaluation)).all()


def test_no_update_with_empty_observations_preserves_frozen_forecast():
    base = np.array([1.0, 0.9, 0.8])
    result = apply_residual_update(base, np.array([0.6, 0.8, 1.0]), pd.DataFrame(), 0.8)

    np.testing.assert_array_equal(result, base)
    assert result is not base


def test_residual_sign_is_observed_minus_base():
    offset, slope = fit_residual_correction([0.8], [0.94], [0.88], 0.8)

    assert np.isclose(offset, 0.01)
    assert slope == 0.0


def test_sparse_selection_uses_rounded_linspace_positions():
    window = pd.DataFrame({"energy_frac": np.arange(7), "SOH": np.arange(7)})

    selected = choose_sparse_update_points(window)

    assert selected.index.tolist() == [0, 2, 4, 6]


def test_sparse_selection_uses_all_points_when_fewer_than_four():
    window = pd.DataFrame({"energy_frac": [0.72, 0.76, 0.80], "SOH": [0.9, 0.8, 0.7]})

    selected = choose_sparse_update_points(window)

    assert selected.index.tolist() == [0, 1, 2]


def test_regularized_residual_fit_matches_direct_linear_algebra():
    x = np.array([0.72, 0.76, 0.80])
    observed = np.array([0.94, 0.91, 0.89])
    base = np.array([0.92, 0.90, 0.88])
    z = (x - 0.8) / 0.1
    design = np.column_stack([np.ones_like(z), z])
    residual = observed - base
    expected = (len(x) / (len(x) + 4.0)) * np.linalg.solve(
        design.T @ design + np.diag([0.35, 2.50]),
        design.T @ residual,
    )

    actual = fit_residual_correction(x, observed, base, 0.8)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_update_applies_correction_only_after_cutoff():
    x = np.array([0.70, 0.75, 0.80, 0.90, 1.00])
    base = np.array([1.00, 0.98, 0.96, 0.94, 0.92])
    points = pd.DataFrame({"energy_frac": [0.80], "SOH": [1.02]})

    updated = apply_residual_update(base, x, points, 0.80)

    np.testing.assert_allclose(updated, [1.00, 0.98, 0.96, 0.95, 0.93])


def test_update_clips_correction_and_enforces_monotonic_soh_range():
    x = np.array([0.7, 0.8, 0.9])
    base = np.array([1.2, 1.1, 0.5])
    points = pd.DataFrame({"energy_frac": [0.8], "SOH": [2.0]})

    updated = apply_residual_update(base, x, points, 0.8)

    np.testing.assert_allclose(updated, [1.05, 1.05, 0.56])


def test_rmse_uses_only_evaluation_region(monkeypatch):
    table = synthetic_cycles()
    base = table["SOH"].to_numpy().copy()
    base[table["energy_frac"].to_numpy() <= 0.80] += 0.5
    monkeypatch.setattr(observation, "evaluate_target", lambda *_: {"fleet_pred": base})

    result = evaluate_observation_update(table, "BTEST")
    no_update = result["metrics"].query("case == 'no_update'").iloc[0]

    assert no_update["rmse_soh"] == 0.0


def test_sparse_and_dense_strategies_remain_distinct(monkeypatch):
    x = np.round(np.arange(0.0, 1.001, 0.02), 2)
    table = pd.DataFrame(
        {
            "battery_id": "BTEST",
            "discharge_index": np.arange(len(x)),
            "energy_frac": x,
            "SOH": 1.0 - 0.2 * x,
        }
    )
    x = table["energy_frac"].to_numpy()
    table.loc[(table["energy_frac"] > 0.70) & (table["energy_frac"] < 0.80), "SOH"] += 0.03
    base = 1.0 - 0.18 * x
    monkeypatch.setattr(observation, "evaluate_target", lambda *_: {"fleet_pred": base})

    result = evaluate_observation_update(table, "BTEST")

    assert len(result["sparse_update"]) == 4
    assert len(result["full_window_update"]) == 5
    assert not np.array_equal(result["pred_sparse"], result["pred_full_window"])


def test_split_rejects_too_few_points():
    table = pd.DataFrame({"energy_frac": [0.1, 0.8]})

    try:
        split_by_energy_fraction(table)
    except ValueError as error:
        assert "Not enough history/future points" in str(error)
    else:
        raise AssertionError("Expected split validation to fail")
