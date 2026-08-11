"""Deterministic post-forecast observation updates for capacity aging.

The frozen capacity forecast comes from the base capacity workflow. Target
observations in the update window calibrate a residual correction, while model
quality is measured separately in the later evaluation region.
"""

import numpy as np
import pandas as pd

from .capacity import evaluate_target
from .metrics import mae, rmse


SPLIT_X = 0.70
FULL_WINDOW_CUTOFF_X = 0.80
SPARSE_UPDATE_COUNT = 4
SOH_RANGE = (0.4, 1.05)


def split_by_energy_fraction(table):
    """Split cycles into history at or below 0.70 and later future cycles."""
    table = table.sort_values("energy_frac").reset_index(drop=True)
    history = table[table["energy_frac"] <= SPLIT_X].copy()
    future = table[table["energy_frac"] > SPLIT_X].copy()
    if len(history) < 12 or len(future) < 6:
        raise ValueError("Not enough history/future points after the 0.70 split.")
    return history, future


def choose_sparse_update_points(update_window):
    """Select up to four evenly spaced rows from the observation window."""
    early = update_window.copy()
    positions = np.linspace(0, len(early) - 1, min(SPARSE_UPDATE_COUNT, len(early)))
    indices = sorted(set(int(round(position)) for position in positions))
    return early.iloc[indices].copy()


def fit_residual_correction(update_x, update_y, base_update_pred, cutoff_x):
    """Fit the Example-18 regularized offset-and-slope residual correction."""
    update_x = np.asarray(update_x, dtype=float)
    residual = np.asarray(update_y, dtype=float) - np.asarray(base_update_pred, dtype=float)
    z = (update_x - float(cutoff_x)) / 0.10
    design = np.column_stack([np.ones_like(z), z])

    n_points = len(update_x)
    if n_points == 0:
        return 0.0, 0.0
    if n_points == 1:
        shrink = n_points / (n_points + 5.0)
        return float(shrink * residual[0]), 0.0

    penalty = np.diag([0.35, 2.50])
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
    shrink = n_points / (n_points + 4.0)
    return float(shrink * solution[0]), float(shrink * solution[1])


def apply_residual_update(base_pred, x_full, update_points, update_cutoff_x):
    """Apply the fitted residual correction after the observation cutoff.

    The base forecast is copied unchanged when there are no observations.
    Otherwise, the correction is clipped, then the complete prediction is
    constrained to be non-increasing and to remain within the SOH range.
    """
    updated = np.asarray(base_pred, dtype=float).copy()
    if update_points.empty:
        return updated

    update_x = update_points["energy_frac"].to_numpy(dtype=float)
    update_y = update_points["SOH"].to_numpy(dtype=float)
    base_update = np.interp(update_x, x_full, base_pred)
    offset, slope = fit_residual_correction(
        update_x,
        update_y,
        base_update,
        update_cutoff_x,
    )

    future_mask = x_full > update_cutoff_x
    z = (x_full[future_mask] - float(update_cutoff_x)) / 0.10
    correction = offset + slope * z
    max_abs_correction = 0.060
    correction = np.clip(correction, -max_abs_correction, max_abs_correction)
    updated[future_mask] = base_pred[future_mask] + correction

    order = np.argsort(x_full)
    ordered = updated[order]
    constrained = np.minimum.accumulate(ordered)
    constrained = np.clip(constrained, SOH_RANGE[0], SOH_RANGE[1])
    restored = np.empty_like(constrained)
    restored[order] = constrained
    return restored


def _metric_row(battery_id, case, update_start_x, update_end_x, history, update_points, eval_points, pred):
    y_eval = eval_points["SOH"].to_numpy(dtype=float)
    pred_eval = pred[eval_points.index.to_numpy()]
    return {
        "battery_id": battery_id,
        "case": case,
        "update_start_x": update_start_x,
        "update_end_x": update_end_x,
        "n_history_points": int(len(history)),
        "n_update_points": int(len(update_points)),
        "n_eval_points": int(len(eval_points)),
        "rmse_soh": rmse(y_eval, pred_eval),
        "mae_soh": mae(y_eval, pred_eval),
    }


def _add_matched_no_update_improvement(rows, full, base_pred):
    for row in rows:
        eval_points = full[full["energy_frac"] > FULL_WINDOW_CUTOFF_X].copy()
        y_eval = eval_points["SOH"].to_numpy(dtype=float)
        base_eval = base_pred[eval_points.index.to_numpy()]
        matched_no_update_rmse = rmse(y_eval, base_eval)
        row["matched_no_update_rmse_soh"] = matched_no_update_rmse
        row["improvement_vs_no_update_percent"] = (
            100.0 * (matched_no_update_rmse - row["rmse_soh"]) / matched_no_update_rmse
            if matched_no_update_rmse > 0.0
            else 0.0
        )
    return rows


def evaluate_observation_update(cycle_dataset, battery_id):
    """Evaluate no, sparse, and dense updates for one target battery.

    The base capacity workflow supplies a frozen final forecast. Observations
    are taken only from ``0.70 < energy_frac <= 0.80``; RMSE and MAE are
    evaluated only for ``energy_frac > 0.80``. The returned dictionary contains
    the split tables, selected observations, predictions, and diagnostics.
    """
    full = (
        cycle_dataset[cycle_dataset["battery_id"] == battery_id]
        .sort_values("energy_frac")
        .reset_index(drop=True)
    )
    history, future = split_by_energy_fraction(full)
    x_full = full["energy_frac"].to_numpy(dtype=float)
    base_pred = evaluate_target(cycle_dataset, battery_id)["fleet_pred"]

    update_window = future[future["energy_frac"] <= FULL_WINDOW_CUTOFF_X].copy()
    if update_window.empty:
        raise ValueError(f"{battery_id} has no observations in the 0.70-0.80 update window.")
    sparse_update = choose_sparse_update_points(update_window)
    sparse_pred = apply_residual_update(
        base_pred,
        x_full,
        sparse_update,
        FULL_WINDOW_CUTOFF_X,
    )

    full_window_pred = apply_residual_update(
        base_pred,
        x_full,
        update_window,
        FULL_WINDOW_CUTOFF_X,
    )

    evaluation = full[full["energy_frac"] > FULL_WINDOW_CUTOFF_X].copy()

    rows = [
        _metric_row(
            battery_id,
            "no_update",
            np.nan,
            FULL_WINDOW_CUTOFF_X,
            history,
            pd.DataFrame(),
            evaluation,
            base_pred,
        ),
        _metric_row(
            battery_id,
            "partial_sparse_update",
            SPLIT_X,
            FULL_WINDOW_CUTOFF_X,
            history,
            sparse_update,
            evaluation,
            sparse_pred,
        ),
        _metric_row(
            battery_id,
            "full_window_update",
            SPLIT_X,
            FULL_WINDOW_CUTOFF_X,
            history,
            update_window,
            evaluation,
            full_window_pred,
        ),
    ]
    rows = _add_matched_no_update_improvement(rows, full, base_pred)

    prediction_table = pd.DataFrame(
        {
            "battery_id": battery_id,
            "normalized_energy_age": full["energy_frac"],
            "discharge_index": full["discharge_index"],
            "measured_soh": full["SOH"],
            "prediction_no_update": base_pred,
            "prediction_partial_sparse_update": sparse_pred,
            "prediction_full_window_update": full_window_pred,
            "region": np.select(
                [
                    full["energy_frac"] <= SPLIT_X,
                    full["energy_frac"] <= FULL_WINDOW_CUTOFF_X,
                ],
                ["history", "update"],
                default="evaluation",
            ),
            "is_sparse_update_point": full.index.isin(sparse_update.index),
        }
    )

    return {
        "battery_id": battery_id,
        "full": full,
        "history": history,
        "future": future,
        "sparse_update": sparse_update,
        "full_window_update": update_window,
        "pred_no_update": base_pred,
        "pred_sparse": sparse_pred,
        "pred_full_window": full_window_pred,
        "metrics": pd.DataFrame(rows),
        "predictions": prediction_table,
    }
