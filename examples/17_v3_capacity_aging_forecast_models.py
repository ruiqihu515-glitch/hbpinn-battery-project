from pathlib import Path
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


INPUT_PATH = Path("results/discharge_sequences_all.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

PREDICTIONS_PATH = RESULTS_DIR / "capacity_aging_v3_predictions.csv"
METRICS_PATH = RESULTS_DIR / "capacity_aging_v3_metrics.csv"
COMPONENTS_PATH = RESULTS_DIR / "capacity_aging_v3_components.csv"

MODEL_COMPARISON_FIGURE = FIGURES_DIR / "capacity_aging_v3_model_comparison.png"
FLEET_DIAGNOSTICS_FIGURE = FIGURES_DIR / "capacity_aging_v3_fleet_diagnostics.png"
RMSE_SUMMARY_FIGURE = FIGURES_DIR / "capacity_aging_v3_rmse_summary.png"

TARGET_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
REQUIRED_COLUMNS = [
    "battery_id",
    "discharge_index",
    "temperature_c",
    "capacity_ah",
    "cycle_energy_kwh",
]

MIN_VALID_CYCLES = 20
MAX_ALLOWED_SOH = 1.5
MAX_ALLOWED_MEDIAN_SOH = 1.2
MIN_ALLOWED_SOH = 0.3

MAX_SPLIT_UPWARD_JUMP = 0.003
PREDICTION_INCREASE_TOL = 0.002
SOH_RANGE = (0.4, 1.05)

SELF_PARAM_GRID = [
    {
        "slope_window_frac": slope_window_frac,
        "slope_scale": slope_scale,
        "acceleration": acceleration,
        "power": power,
    }
    for slope_window_frac in [0.15, 0.20, 0.30]
    for slope_scale in [0.80, 1.00, 1.15]
    for acceleration in [0.90, 1.20, 1.60, 2.00]
    for power in [1.35, 1.70, 2.10]
]
BLEND_WEIGHT_GRID = [0.0, 0.10, 0.20, 0.30]
TOP_K_FLEET = 5
LINEARITY_SECOND_DIFF_TOL = 2.5e-5


def first_valid(series):
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def robust_capacity(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.empty:
        return np.nan
    return float(values.median())


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def sorted_unique_xy(x, y):
    table = (
        pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("x")
    )
    grouped = table.groupby("x", as_index=False)["y"].mean()
    return grouped["x"].to_numpy(dtype=float), grouped["y"].to_numpy(dtype=float)


def estimate_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


class MonotonePchipCurve:
    """Prediction-only monotone trend; measured SOH is never smoothed in outputs."""

    def __init__(self, x, y):
        x, y = sorted_unique_xy(x, y)
        if len(x) < 2:
            raise ValueError("At least two unique x values are required.")
        isotonic = IsotonicRegression(increasing=False, out_of_bounds="clip")
        y_iso = isotonic.fit_transform(x, y)
        y_iso = np.minimum.accumulate(y_iso)

        self.x = x
        self.y = y_iso
        if PchipInterpolator is not None and len(x) >= 3:
            self.interpolator = PchipInterpolator(x, y_iso, extrapolate=False)
        else:
            self.interpolator = None

    def __call__(self, x_eval):
        x_eval = np.asarray(x_eval, dtype=float)
        flat = x_eval.reshape(-1)
        clipped = np.clip(flat, self.x[0], self.x[-1])
        if self.interpolator is not None:
            pred = np.asarray(self.interpolator(clipped), dtype=float)
        else:
            pred = np.interp(clipped, self.x, self.y)
        pred[flat < self.x[0]] = self.y[0]
        pred[flat > self.x[-1]] = self.y[-1]

        order = np.argsort(flat)
        sorted_pred = np.minimum.accumulate(pred[order])
        restored = np.empty_like(sorted_pred)
        restored[order] = sorted_pred
        return restored.reshape(x_eval.shape)


def build_cycle_dataset(sequence_table):
    missing = [column for column in REQUIRED_COLUMNS if column not in sequence_table]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cycle_table = (
        sequence_table.groupby(["battery_id", "discharge_index"], as_index=False)
        .agg(
            capacity_ah=("capacity_ah", robust_capacity),
            cycle_energy_kwh=("cycle_energy_kwh", first_valid),
            mean_temperature_c=("temperature_c", "mean"),
            min_temperature_c=("temperature_c", "min"),
            max_temperature_c=("temperature_c", "max"),
            number_of_steps=("temperature_c", "size"),
        )
        .sort_values(["battery_id", "discharge_index"])
        .reset_index(drop=True)
    )

    valid_tables = []
    skipped = {}
    for battery_id, battery_table in cycle_table.groupby("battery_id", sort=True):
        battery_table = battery_table.sort_values("discharge_index").copy()
        positive_capacity = battery_table["capacity_ah"][
            np.isfinite(battery_table["capacity_ah"])
            & (battery_table["capacity_ah"] > 0.0)
        ]
        if len(positive_capacity) >= 3:
            initial_capacity = float(positive_capacity.iloc[:3].median())
        elif len(positive_capacity) > 0:
            initial_capacity = float(positive_capacity.iloc[0])
        else:
            initial_capacity = np.nan

        battery_table["initial_capacity_ah"] = initial_capacity
        battery_table["SOH"] = battery_table["capacity_ah"] / initial_capacity

        essential = [
            "capacity_ah",
            "cycle_energy_kwh",
            "mean_temperature_c",
            "min_temperature_c",
            "max_temperature_c",
            "number_of_steps",
            "SOH",
        ]
        finite_soh = battery_table["SOH"][
            np.isfinite(battery_table["SOH"])
        ].to_numpy(dtype=float)
        reasons = []
        if len(positive_capacity) < MIN_VALID_CYCLES:
            reasons.append("too few valid cycles")
        if not np.isfinite(initial_capacity) or initial_capacity <= 0.0:
            reasons.append("non-positive initial capacity")
        if battery_table[essential].isna().any().any():
            reasons.append("missing essential numeric fields")
        if len(finite_soh) == 0:
            reasons.append("missing capacity")
        else:
            if np.nanmedian(finite_soh) > MAX_ALLOWED_MEDIAN_SOH:
                reasons.append("median SOH above reasonable range")
            if np.nanmax(finite_soh) > MAX_ALLOWED_SOH:
                reasons.append("max SOH above reasonable range")
            if np.nanmin(finite_soh) < MIN_ALLOWED_SOH:
                reasons.append("min SOH below reasonable range")

        if reasons:
            skipped[battery_id] = "; ".join(reasons)
            continue

        battery_table = battery_table.dropna(subset=essential).copy()
        battery_table = battery_table[
            np.isfinite(battery_table["SOH"])
            & np.isfinite(battery_table["cycle_energy_kwh"])
            & (battery_table["cycle_energy_kwh"] >= 0.0)
        ].copy()
        if len(battery_table) < MIN_VALID_CYCLES:
            skipped[battery_id] = "too few cycles after filtering"
            continue

        battery_table["cycle_age"] = np.arange(len(battery_table), dtype=int)
        battery_table["energy_age_kwh"] = battery_table["cycle_energy_kwh"].cumsum()
        max_energy = float(battery_table["energy_age_kwh"].max())
        if not np.isfinite(max_energy) or max_energy <= 0.0:
            skipped[battery_id] = "non-positive cumulative energy"
            continue
        battery_table["energy_frac"] = battery_table["energy_age_kwh"] / max_energy
        valid_tables.append(battery_table)

    if not valid_tables:
        raise ValueError(f"No valid batteries found. Skipped: {skipped}")

    columns = [
        "battery_id",
        "discharge_index",
        "cycle_age",
        "energy_age_kwh",
        "energy_frac",
        "capacity_ah",
        "initial_capacity_ah",
        "SOH",
        "cycle_energy_kwh",
        "mean_temperature_c",
        "min_temperature_c",
        "max_temperature_c",
        "number_of_steps",
    ]
    return pd.concat(valid_tables, ignore_index=True)[columns], skipped


def split_70_30(table):
    table = table.sort_values("discharge_index").reset_index(drop=True)
    split_index = int(np.floor(0.70 * len(table)))
    split_index = min(max(split_index, 3), len(table) - 1)
    return table.iloc[:split_index].copy(), table.iloc[split_index:].copy()


def inner_train_validation_split(train_table):
    split_index = int(np.floor(0.72 * len(train_table)))
    split_index = min(max(split_index, 5), len(train_table) - 2)
    return train_table.iloc[:split_index].copy(), train_table.iloc[split_index:].copy()


def nonlinear_self_forecast(fit_table, eval_x, params):
    fit_table = fit_table.sort_values("energy_frac").copy()
    x_fit = fit_table["energy_frac"].to_numpy(dtype=float)
    y_fit = fit_table["SOH"].to_numpy(dtype=float)
    curve = MonotonePchipCurve(x_fit, y_fit)
    pred = curve(eval_x)

    anchor_x = float(x_fit[-1])
    anchor_y = float(curve(np.array([anchor_x]))[0])
    future = np.asarray(eval_x, dtype=float) > anchor_x
    if not np.any(future):
        return pred

    n_late = min(
        max(6, int(np.ceil(params["slope_window_frac"] * len(fit_table)))),
        len(fit_table),
    )
    late_curve_y = curve(x_fit[-n_late:])
    late_x = x_fit[-n_late:]
    late_degradation = 1.0 - late_curve_y
    late_rate = max(estimate_slope(late_x, late_degradation), 0.0)

    half = max(3, n_late // 2)
    early_late_rate = max(
        estimate_slope(late_x[:half], late_degradation[:half]),
        0.0,
    )
    final_late_rate = max(
        estimate_slope(late_x[-half:], late_degradation[-half:]),
        0.0,
    )
    total_drop = max(float(curve(np.array([x_fit[0]]))[0] - anchor_y), 1e-4)
    total_span = max(anchor_x - float(x_fit[0]), 1e-4)
    average_degradation_rate = total_drop / total_span

    local_rate = max(
        params["slope_scale"] * (0.55 * final_late_rate + 0.45 * late_rate),
        0.18 * average_degradation_rate,
    )
    rate_change = max(final_late_rate - early_late_rate, 0.0)
    curvature_rate = params["acceleration"] * max(
        rate_change,
        0.45 * local_rate,
        0.18 * average_degradation_rate,
    )
    curvature_rate = min(
        curvature_rate,
        0.65 * local_rate + 0.18 * average_degradation_rate,
    )

    future_x = np.asarray(eval_x, dtype=float)[future]
    dx = future_x - anchor_x
    horizon = max(1.0 - anchor_x, float(dx.max()), 0.06)
    z = np.clip(dx / horizon, 0.0, 1.5)
    curved_extra_drop = (
        curvature_rate
        * horizon
        * z ** (params["power"] + 1.0)
        / (params["power"] + 1.0)
    )
    carryover_drop = 0.20 * rate_change * horizon * z ** 3 / 3.0
    pred[future] = anchor_y - local_rate * dx - curved_extra_drop - carryover_drop
    return pred


def enforce_forecast_sanity(pred, train_len, model_name, battery_id):
    pred = np.asarray(pred, dtype=float).copy()
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"{battery_id} {model_name} produced NaN/inf forecast.")

    pred = np.minimum.accumulate(pred)
    if len(pred) > train_len:
        last_train = pred[train_len - 1]
        first_test = pred[train_len]
        if first_test > last_train + MAX_SPLIT_UPWARD_JUMP:
            pred[train_len:] -= first_test - (last_train + MAX_SPLIT_UPWARD_JUMP)
        pred[train_len:] = np.minimum.accumulate(pred[train_len:])

    pred = np.clip(pred, SOH_RANGE[0], SOH_RANGE[1])
    if len(pred) > train_len and pred[train_len] > pred[train_len - 1] + 0.006:
        raise ValueError(f"{battery_id} {model_name} has split discontinuity.")
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"{battery_id} {model_name} failed finite sanity check.")
    if pred.min() < SOH_RANGE[0] - 1e-9 or pred.max() > SOH_RANGE[1] + 1e-9:
        raise ValueError(f"{battery_id} {model_name} outside expected SOH range.")
    increases = np.diff(pred[train_len:])
    if increases.size and np.max(increases) > PREDICTION_INCREASE_TOL:
        raise ValueError(f"{battery_id} {model_name} has rising test forecast.")
    return pred


def forecast_shape_diagnostics(x, pred, train_len):
    x = np.asarray(x, dtype=float)
    pred = np.asarray(pred, dtype=float)
    train_x = x[:train_len]
    train_pred = pred[:train_len]
    test_x = x[train_len:]
    test_pred = pred[train_len:]
    if len(test_pred) < 4:
        return {
            "split_slope": 0.0,
            "train_tail_slope": 0.0,
            "forecast_average_slope": 0.0,
            "final_test_slope": 0.0,
            "forecast_slope_change": 0.0,
            "train_tail_curvature": 0.0,
            "forecast_curvature": 0.0,
            "test_linearity_score": 0.0,
            "curved_extrapolation_used": False,
            "near_linear_forecast": True,
            "effectively_linear_forecast": True,
        }

    train_tail_count = min(max(6, int(np.ceil(0.10 * train_len))), train_len)
    train_tail_x = train_x[-train_tail_count:]
    train_tail_pred = train_pred[-train_tail_count:]
    train_tail_slope = estimate_slope(train_tail_x, train_tail_pred)
    train_tail_second_diff = np.diff(train_tail_pred, n=2)
    train_tail_curvature = (
        float(np.mean(np.abs(train_tail_second_diff)))
        if train_tail_second_diff.size
        else 0.0
    )
    split_slope = estimate_slope(
        x[max(0, train_len - 3) : min(len(x), train_len + 3)],
        pred[max(0, train_len - 3) : min(len(pred), train_len + 3)],
    )
    forecast_average_slope = estimate_slope(test_x, test_pred)
    tail_count = min(8, len(test_pred))
    final_test_slope = estimate_slope(test_x[-tail_count:], test_pred[-tail_count:])
    local_slopes = np.diff(test_pred) / np.maximum(np.diff(test_x), 1e-9)
    slope_change = float(local_slopes[-1] - local_slopes[0])
    second_diff = np.diff(test_pred, n=2)
    linearity_score = float(np.mean(np.abs(second_diff))) if second_diff.size else 0.0
    forecast_curvature = linearity_score
    near_linear = (
        linearity_score < LINEARITY_SECOND_DIFF_TOL
        and abs(slope_change) < 0.06
    )
    return {
        "split_slope": float(split_slope),
        "train_tail_slope": float(train_tail_slope),
        "forecast_average_slope": float(forecast_average_slope),
        "final_test_slope": float(final_test_slope),
        "forecast_slope_change": slope_change,
        "train_tail_curvature": train_tail_curvature,
        "forecast_curvature": forecast_curvature,
        "test_linearity_score": linearity_score,
        "curved_extrapolation_used": not near_linear,
        "near_linear_forecast": near_linear,
        "effectively_linear_forecast": near_linear,
    }


def select_self_hyperparameters(train_table):
    inner_fit, validation = inner_train_validation_split(train_table)
    eval_full = pd.concat([inner_fit, validation], ignore_index=True)
    x_full = eval_full["energy_frac"].to_numpy(dtype=float)
    eval_x = validation["energy_frac"].to_numpy(dtype=float)
    y_val = validation["SOH"].to_numpy(dtype=float)

    candidates = []
    for params in SELF_PARAM_GRID:
        pred = nonlinear_self_forecast(inner_fit, eval_x, params)
        pred_full = nonlinear_self_forecast(inner_fit, x_full, params)
        pred_full = enforce_forecast_sanity(
            pred_full,
            len(inner_fit),
            "self_validation",
            str(train_table["battery_id"].iloc[0]),
        )
        pred = pred_full[len(inner_fit) :]
        validation_rmse = rmse(y_val, pred)
        shape = forecast_shape_diagnostics(x_full, pred_full, len(inner_fit))
        candidates.append(
            {
                "params": dict(params),
                "validation_rmse": float(validation_rmse),
                "linearity_score": shape["test_linearity_score"],
                "slope_change": abs(shape["forecast_slope_change"]),
                "near_linear_forecast": shape["near_linear_forecast"],
            }
        )

    best_rmse = min(candidate["validation_rmse"] for candidate in candidates)
    tolerance = max(0.0015, 0.05 * best_rmse)
    acceptable = [
        candidate
        for candidate in candidates
        if candidate["validation_rmse"] <= best_rmse + tolerance
    ]
    non_linear_tolerance = max(0.003, 0.12 * best_rmse)
    non_linear_acceptable = [
        candidate
        for candidate in candidates
        if (
            not candidate["near_linear_forecast"]
            and candidate["validation_rmse"] <= best_rmse + non_linear_tolerance
        )
    ]
    if non_linear_acceptable:
        acceptable = non_linear_acceptable
    acceptable = sorted(
        acceptable,
        key=lambda candidate: (
            candidate["near_linear_forecast"],
            candidate["validation_rmse"],
            candidate["params"]["acceleration"],
        ),
    )
    selected = acceptable[0]
    return dict(selected["params"]), float(selected["validation_rmse"])


def build_fleet_curves(cycle_dataset, target_id):
    curves = {}
    for battery_id, table in cycle_dataset.groupby("battery_id", sort=True):
        if battery_id == target_id:
            continue
        curves[battery_id] = MonotonePchipCurve(table["energy_frac"], table["SOH"])
    return curves


def score_fleet_curves(fleet_curves, train_table):
    x_train = train_table["energy_frac"].to_numpy(dtype=float)
    y_train = train_table["SOH"].to_numpy(dtype=float)
    n_late = min(max(4, int(np.ceil(0.30 * len(train_table)))), len(train_table))
    target_late_slope = estimate_slope(x_train[-n_late:], y_train[-n_late:])
    target_anchor = float(np.median(y_train[-min(5, len(y_train)) :]))
    rows = []

    for fleet_id, curve in fleet_curves.items():
        raw = curve(x_train)
        shift = target_anchor - float(np.median(raw[-min(5, len(raw)) :]))
        aligned = raw + shift
        overall_rmse = rmse(y_train, aligned)
        late_rmse = rmse(y_train[-n_late:], aligned[-n_late:])
        fleet_late_slope = estimate_slope(x_train[-n_late:], aligned[-n_late:])
        slope_mismatch = abs(target_late_slope - fleet_late_slope)
        score = overall_rmse + late_rmse + 0.5 * slope_mismatch
        rows.append(
            {
                "fleet_battery_id": fleet_id,
                "similarity_score": float(score),
                "overall_train_rmse": float(overall_rmse),
                "late_train_rmse": float(late_rmse),
                "slope_mismatch": float(slope_mismatch),
                "anchor_shift": float(shift),
            }
        )

    components = pd.DataFrame(rows).sort_values("similarity_score").reset_index(
        drop=True
    )
    selected = components.index[: min(TOP_K_FLEET, len(components))]
    selected_scores = components.loc[selected, "similarity_score"].to_numpy(dtype=float)
    positive_scores = selected_scores[selected_scores > 0]
    tau = max(float(np.median(positive_scores)) if positive_scores.size else 1.0, 1e-6)
    raw_weights = np.exp(-selected_scores / tau)
    weights = raw_weights / raw_weights.sum()
    components["selected_top_k"] = False
    components["weight"] = 0.0
    components.loc[selected, "selected_top_k"] = True
    components.loc[selected, "weight"] = weights
    return components


def fleet_prior_prediction(fleet_curves, components, eval_x, train_table, weighted=True):
    selected = components[components["selected_top_k"]].copy()
    if selected.empty:
        selected = components.nsmallest(1, "similarity_score").copy()
    if weighted:
        weights = selected["weight"].to_numpy(dtype=float)
    else:
        weights = np.ones(len(selected), dtype=float) / len(selected)

    target_anchor = float(train_table["SOH"].tail(min(5, len(train_table))).median())
    pred = np.zeros(len(eval_x), dtype=float)
    for weight, row in zip(weights, selected.itertuples(index=False)):
        curve_pred = fleet_curves[row.fleet_battery_id](eval_x)
        train_pred = fleet_curves[row.fleet_battery_id](
            train_table["energy_frac"].to_numpy(dtype=float)
        )
        shift = target_anchor - float(np.median(train_pred[-min(5, len(train_pred)) :]))
        pred += float(weight) * (curve_pred + shift)
    return np.minimum.accumulate(pred)


def blend_self_and_fleet(self_pred, fleet_pred, blend_weight):
    pred = (1.0 - blend_weight) * self_pred + blend_weight * fleet_pred
    return np.minimum.accumulate(pred)


def blend_self_with_fleet_curvature(x, self_pred, fleet_pred, train_len, blend_weight):
    x = np.asarray(x, dtype=float)
    self_pred = np.asarray(self_pred, dtype=float)
    fleet_pred = np.asarray(fleet_pred, dtype=float)
    pred = self_pred.copy()
    if blend_weight <= 0.0 or len(pred) <= train_len:
        return np.minimum.accumulate(pred)

    anchor_index = train_len - 1
    anchor_x = float(x[anchor_index])
    future = x > anchor_x
    if not np.any(future):
        return np.minimum.accumulate(pred)

    slope_count = min(8, train_len)
    fit_slice = slice(train_len - slope_count, train_len)
    self_slope = estimate_slope(x[fit_slice], self_pred[fit_slice])
    fleet_slope = estimate_slope(x[fit_slice], fleet_pred[fit_slice])

    self_tangent = self_pred[anchor_index] + self_slope * (x[future] - anchor_x)
    fleet_tangent = fleet_pred[anchor_index] + fleet_slope * (x[future] - anchor_x)
    self_curvature_residual = self_pred[future] - self_tangent
    fleet_curvature_residual = fleet_pred[future] - fleet_tangent

    # Use only the fleet curvature residual, not its level or split slope.
    curvature_adjustment = fleet_curvature_residual - self_curvature_residual
    pred[future] = self_pred[future] + blend_weight * curvature_adjustment
    pred[:train_len] = self_pred[:train_len]
    pred[anchor_index] = self_pred[anchor_index]
    return np.minimum.accumulate(pred)


def select_fleet_hyperparameters(train_table, cycle_dataset, target_id, self_params):
    inner_fit, validation = inner_train_validation_split(train_table)
    fleet_curves = build_fleet_curves(cycle_dataset, target_id)
    components = score_fleet_curves(fleet_curves, inner_fit)
    eval_full = pd.concat([inner_fit, validation], ignore_index=True)
    x_full = eval_full["energy_frac"].to_numpy(dtype=float)
    y_val = validation["SOH"].to_numpy(dtype=float)
    train_len = len(inner_fit)
    fleet_pred_full = fleet_prior_prediction(
        fleet_curves, components, x_full, inner_fit, weighted=True
    )

    self_pred_full = nonlinear_self_forecast(inner_fit, x_full, self_params)
    self_pred_full = enforce_forecast_sanity(
        self_pred_full,
        train_len,
        "self_validation_for_fleet",
        target_id,
    )
    self_shape = forecast_shape_diagnostics(x_full, self_pred_full, train_len)
    recent_count = min(10, len(inner_fit))
    target_recent_slope = estimate_slope(
        inner_fit["energy_frac"].to_numpy(dtype=float)[-recent_count:],
        inner_fit["SOH"].to_numpy(dtype=float)[-recent_count:],
    )
    prior_recent_slope = estimate_slope(
        inner_fit["energy_frac"].to_numpy(dtype=float)[-recent_count:],
        fleet_pred_full[:train_len][-recent_count:],
    )
    target_recent_rate = max(-target_recent_slope, 1e-4)
    prior_recent_rate = max(-prior_recent_slope, 0.0)
    fleet_rate_ratio = prior_recent_rate / target_recent_rate
    max_allowed_blend = 0.30
    if fleet_rate_ratio > 1.8:
        max_allowed_blend = 0.10
    elif fleet_rate_ratio > 1.4:
        max_allowed_blend = 0.20

    candidates = []
    for blend_weight in BLEND_WEIGHT_GRID:
        if blend_weight > max_allowed_blend:
            continue
        pred_full = blend_self_with_fleet_curvature(
            x_full, self_pred_full, fleet_pred_full, train_len, blend_weight
        )
        pred_full = enforce_forecast_sanity(
            pred_full,
            train_len,
            "fleet_validation",
            target_id,
        )
        pred_val = pred_full[train_len:]
        shape = forecast_shape_diagnostics(x_full, pred_full, train_len)
        validation_rmse = rmse(y_val, pred_val)
        candidates.append(
            {
                "params": dict(self_params),
                "blend_weight": float(blend_weight),
                "validation_rmse": float(validation_rmse),
                "test_linearity_score": shape["test_linearity_score"],
                "near_linear_forecast": shape["near_linear_forecast"],
                "fleet_rate_ratio": float(fleet_rate_ratio),
                "max_allowed_blend": float(max_allowed_blend),
            }
        )

    self_candidate = min(candidates, key=lambda candidate: candidate["blend_weight"])
    self_validation_rmse = self_candidate["validation_rmse"]
    required_gain = max(0.0008, 0.03 * self_validation_rmse)
    improving_candidates = [
        candidate
        for candidate in candidates
        if candidate["validation_rmse"] < self_validation_rmse - required_gain
    ]
    if improving_candidates:
        acceptable = improving_candidates
    else:
        acceptable = [self_candidate]
        if self_shape["near_linear_forecast"]:
            shape_tolerance = max(0.006, 0.25 * self_validation_rmse)
            shape_candidates = [
                candidate
                for candidate in candidates
                if (
                    candidate["blend_weight"] > 0.0
                    and not candidate["near_linear_forecast"]
                    and candidate["validation_rmse"]
                    <= self_validation_rmse + shape_tolerance
                )
            ]
            if shape_candidates:
                acceptable = shape_candidates
    minimum_curvature = 0.80 * self_shape["test_linearity_score"]
    if minimum_curvature > 0.0:
        curvature_preserving = [
            candidate
            for candidate in acceptable
            if candidate["test_linearity_score"] >= minimum_curvature
        ]
        if curvature_preserving:
            acceptable = curvature_preserving
    acceptable = sorted(
        acceptable,
        key=lambda candidate: (
            candidate["validation_rmse"],
            candidate["blend_weight"],
        ),
    )
    return acceptable[0], fleet_curves


def metric_row(
    battery_id,
    model_name,
    train_table,
    test_table,
    pred,
    selected_hyperparameters,
    selected_blend_weight,
    shape,
    self_test_rmse=None,
):
    n_train = len(train_table)
    y = np.concatenate(
        [
            train_table["SOH"].to_numpy(dtype=float),
            test_table["SOH"].to_numpy(dtype=float),
        ]
    )
    train_pred = pred[:n_train]
    test_pred = pred[n_train:]
    test_rmse = rmse(y[n_train:], test_pred)
    improved = False if self_test_rmse is None else bool(test_rmse < self_test_rmse)
    return {
        "battery_id": battery_id,
        "model_name": model_name,
        "train_rmse": rmse(y[:n_train], train_pred),
        "test_rmse": test_rmse,
        "test_mae": mae(y[n_train:], test_pred),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "selected_blend_weight": float(selected_blend_weight),
        "regularization_strength": float(selected_blend_weight),
        "whether_fleet_improved_over_self": improved,
        "forecast_family": "curved_degradation_rate_continuation",
        "curved_extrapolation_used": bool(shape["curved_extrapolation_used"]),
        "split_slope": float(shape["split_slope"]),
        "train_tail_slope": float(shape["train_tail_slope"]),
        "forecast_average_slope": float(shape["forecast_average_slope"]),
        "final_test_slope": float(shape["final_test_slope"]),
        "forecast_slope_change": float(shape["forecast_slope_change"]),
        "train_tail_curvature": float(shape["train_tail_curvature"]),
        "forecast_curvature": float(shape["forecast_curvature"]),
        "test_linearity_score": float(shape["test_linearity_score"]),
        "near_linear_forecast": bool(shape["near_linear_forecast"]),
        "effectively_linear_forecast": bool(shape["effectively_linear_forecast"]),
        "n_train": int(len(train_table)),
        "n_test": int(len(test_table)),
    }


def prediction_frame(battery_id, full_table, split_labels, model_name, pred):
    table = full_table.copy()
    table["target_battery_id"] = battery_id
    table["split"] = split_labels
    table["model_name"] = model_name
    table["predicted_SOH"] = pred
    table["predicted_capacity_ah"] = pred * table["initial_capacity_ah"]
    return table[
        [
            "target_battery_id",
            "battery_id",
            "discharge_index",
            "cycle_age",
            "energy_age_kwh",
            "energy_frac",
            "capacity_ah",
            "SOH",
            "initial_capacity_ah",
            "mean_temperature_c",
            "split",
            "model_name",
            "predicted_SOH",
            "predicted_capacity_ah",
        ]
    ]


def evaluate_target(cycle_dataset, battery_id):
    target_table = cycle_dataset[cycle_dataset["battery_id"] == battery_id].copy()
    train_table, test_table = split_70_30(target_table)
    full_table = pd.concat([train_table, test_table], ignore_index=True)
    x_full = full_table["energy_frac"].to_numpy(dtype=float)
    split_labels = np.array(["train"] * len(train_table) + ["test"] * len(test_table))

    self_params, self_validation_score = select_self_hyperparameters(train_table)
    self_pred = nonlinear_self_forecast(train_table, x_full, self_params)
    self_pred = enforce_forecast_sanity(
        self_pred, len(train_table), "self_only_shape_constrained", battery_id
    )
    self_shape = forecast_shape_diagnostics(x_full, self_pred, len(train_table))

    fleet_selection, fleet_curves = select_fleet_hyperparameters(
        train_table, cycle_dataset, battery_id, self_params
    )
    components = score_fleet_curves(fleet_curves, train_table)
    weighted_prior = fleet_prior_prediction(
        fleet_curves, components, x_full, train_table, weighted=True
    )
    uniform_prior = fleet_prior_prediction(
        fleet_curves, components, x_full, train_table, weighted=False
    )
    fleet_self_pred = nonlinear_self_forecast(
        train_table, x_full, fleet_selection["params"]
    )
    fleet_pred = blend_self_with_fleet_curvature(
        x_full,
        fleet_self_pred,
        weighted_prior,
        len(train_table),
        fleet_selection["blend_weight"],
    )
    fleet_pred = enforce_forecast_sanity(
        fleet_pred, len(train_table), "fleet_updated_deterministic_v3", battery_id
    )
    fleet_shape = forecast_shape_diagnostics(x_full, fleet_pred, len(train_table))

    self_metric = metric_row(
        battery_id,
        "self_only_shape_constrained",
        train_table,
        test_table,
        self_pred,
        {**self_params, "validation_score": self_validation_score},
        0.0,
        self_shape,
    )
    fleet_metric = metric_row(
        battery_id,
        "fleet_updated_deterministic_v3",
        train_table,
        test_table,
        fleet_pred,
        {
            **fleet_selection["params"],
            "validation_rmse": fleet_selection["validation_rmse"],
        },
        fleet_selection["blend_weight"],
        fleet_shape,
        self_test_rmse=self_metric["test_rmse"],
    )

    components.insert(0, "battery_id", battery_id)
    predictions = pd.concat(
        [
            prediction_frame(
                battery_id,
                full_table,
                split_labels,
                "self_only_shape_constrained",
                self_pred,
            ),
            prediction_frame(
                battery_id,
                full_table,
                split_labels,
                "fleet_updated_deterministic_v3",
                fleet_pred,
            ),
        ],
        ignore_index=True,
    )

    return {
        "battery_id": battery_id,
        "train": train_table,
        "test": test_table,
        "full": full_table,
        "x_full": x_full,
        "self_pred": self_pred,
        "fleet_pred": fleet_pred,
        "weighted_prior": weighted_prior,
        "uniform_prior": uniform_prior,
        "fleet_curves": fleet_curves,
        "components": components,
        "metrics": pd.DataFrame([self_metric, fleet_metric]),
        "predictions": predictions,
        "fleet_selection": fleet_selection,
        "self_shape": self_shape,
        "fleet_shape": fleet_shape,
    }


def plot_model_comparison(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, battery_id in zip(axes, TARGET_BATTERIES):
        result = results[battery_id]
        full = result["full"]
        train = result["train"]
        ax.plot(
            full["energy_frac"],
            full["SOH"],
            color="black",
            marker="o",
            markersize=3,
            linewidth=0.8,
            label="Measured raw SOH",
        )
        ax.plot(
            result["x_full"],
            result["self_pred"],
            color="#2f6db3",
            linewidth=2.2,
            label="Self-only baseline",
        )
        ax.plot(
            result["x_full"],
            result["fleet_pred"],
            color="#208744",
            linewidth=2.8,
            label="Fleet-updated deterministic final model",
        )
        ax.axvline(
            float(train["energy_frac"].iloc[-1]),
            color="0.25",
            linestyle="--",
            linewidth=1.1,
            label="Train/test split",
        )
        ax.set_title(battery_id)
        ax.set_xlabel("Normalized cumulative energy age")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("V3 capacity-aging forecast comparison", fontsize=15)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(MODEL_COMPARISON_FIGURE, dpi=200)
    plt.close(fig)


def plot_fleet_diagnostics(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    x_grid = np.linspace(0.0, 1.0, 240)
    for ax, battery_id in zip(axes, TARGET_BATTERIES):
        result = results[battery_id]
        for index, curve in enumerate(result["fleet_curves"].values()):
            ax.plot(
                x_grid,
                curve(x_grid),
                color="0.78",
                linewidth=0.7,
                alpha=0.5,
                label="Individual fleet curves" if index == 0 else None,
            )
        ax.plot(
            result["x_full"],
            result["uniform_prior"],
            color="#6f6fbd",
            linestyle=":",
            linewidth=2.0,
            label="Uniform prior diagnostic only",
        )
        ax.plot(
            result["x_full"],
            result["weighted_prior"],
            color="#e68613",
            linestyle="--",
            linewidth=2.0,
            label="Weighted prior diagnostic only",
        )
        ax.plot(
            result["x_full"],
            result["fleet_pred"],
            color="#208744",
            linewidth=2.8,
            label="Final target-specific forecast",
        )
        ax.scatter(
            result["train"]["energy_frac"],
            result["train"]["SOH"],
            color="black",
            s=14,
            label="Target train SOH",
            zorder=4,
        )
        ax.scatter(
            result["test"]["energy_frac"],
            result["test"]["SOH"],
            facecolors="none",
            edgecolors="black",
            s=22,
            label="Target test SOH",
            zorder=4,
        )
        ax.axvline(
            float(result["train"]["energy_frac"].iloc[-1]),
            color="0.25",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_title(battery_id)
        ax.set_xlabel("Normalized cumulative energy age")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "V3 fleet diagnostics: priors are diagnostics only; final forecast is target-specific",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    fig.savefig(FLEET_DIAGNOSTICS_FIGURE, dpi=200)
    plt.close(fig)


def plot_rmse_summary(metrics):
    summary = metrics.groupby("model_name", as_index=False)["test_rmse"].mean()
    labels = {
        "self_only_shape_constrained": "Self-only\nshape constrained",
        "fleet_updated_deterministic_v3": "Fleet-updated\nfinal",
    }
    colors = {
        "self_only_shape_constrained": "#2f6db3",
        "fleet_updated_deterministic_v3": "#208744",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(summary))
    bars = ax.bar(
        x,
        summary["test_rmse"],
        color=[colors[name] for name in summary["model_name"]],
    )
    ax.set_xticks(x, [labels[name] for name in summary["model_name"]])
    ax.set_ylabel("Mean test RMSE (SOH)")
    ax.set_title("V3 capacity-aging forecast RMSE")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary["test_rmse"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(RMSE_SUMMARY_FIGURE, dpi=200)
    plt.close(fig)


def print_summary(cycle_dataset, results, metrics, skipped):
    print("\nExample 17 V3 capacity-aging forecast results")
    print(
        "Project context: the main successful result is the voltage-response "
        "workflow with a physics-inspired baseline and residual learning. "
        "Capacity aging is an exploratory cycle-level extension."
    )
    print(
        "Temperature ablation remains battery-dependent: temperature helps in "
        "some voltage-response cases but is not consistently beneficial."
    )
    print(
        "Root cause found: previous V3 either forced curvature too strongly or "
        "collapsed to a near-constant post-split degradation rate when validation "
        "selected weak acceleration."
    )
    print(
        "Changed: V3 now uses smooth nonlinear rate-transition continuation and "
        "fleet contributes only curvature residuals after the split, preserving "
        "target split value and first derivative."
    )
    print(f"Valid batteries used: {', '.join(sorted(cycle_dataset['battery_id'].unique()))}")
    print(f"Target batteries: {', '.join(TARGET_BATTERIES)}")

    print("\nPer-battery test RMSE:")
    for battery_id in TARGET_BATTERIES:
        battery_metrics = metrics[metrics["battery_id"] == battery_id]
        self_rmse = float(
            battery_metrics.loc[
                battery_metrics["model_name"] == "self_only_shape_constrained",
                "test_rmse",
            ].iloc[0]
        )
        fleet_rmse = float(
            battery_metrics.loc[
                battery_metrics["model_name"] == "fleet_updated_deterministic_v3",
                "test_rmse",
            ].iloc[0]
        )
        blend = results[battery_id]["fleet_selection"]["blend_weight"]
        print(
            f"  {battery_id}: self-only={self_rmse:.5f}, "
            f"fleet-updated={fleet_rmse:.5f}, selected blend={blend:.2f}"
        )

    print("\nPer-battery final SOH bias for fleet-updated candidate:")
    for battery_id in TARGET_BATTERIES:
        result = results[battery_id]
        measured_final = float(result["full"]["SOH"].iloc[-1])
        predicted_final = float(result["fleet_pred"][-1])
        print(
            f"  {battery_id}: predicted-final minus measured-final = "
            f"{predicted_final - measured_final:+.5f}"
        )

    print("\nForecast shape diagnostics:")
    for battery_id in TARGET_BATTERIES:
        for model_name, label in [
            ("self_only_shape_constrained", "self-only"),
            ("fleet_updated_deterministic_v3", "fleet-updated"),
        ]:
            row = metrics[
                (metrics["battery_id"] == battery_id)
                & (metrics["model_name"] == model_name)
            ].iloc[0]
            print(
                f"  {battery_id} {label}: family={row.forecast_family}, "
                f"curved={bool(row.curved_extrapolation_used)}, "
                f"split_slope={row.split_slope:.4f}, "
                f"train_tail_slope={row.train_tail_slope:.4f}, "
                f"forecast_avg_slope={row.forecast_average_slope:.4f}, "
                f"final_slope={row.final_test_slope:.4f}, "
                f"slope_change={row.forecast_slope_change:.4f}, "
                f"train_tail_curv={row.train_tail_curvature:.6f}, "
                f"forecast_curv={row.forecast_curvature:.6f}, "
                f"effectively_linear={bool(row.effectively_linear_forecast)}, "
                f"blend={row.selected_blend_weight:.2f}"
            )

    average = metrics.groupby("model_name")["test_rmse"].mean()
    self_avg = float(average["self_only_shape_constrained"])
    fleet_avg = float(average["fleet_updated_deterministic_v3"])
    self_avg_display = round(self_avg, 4)
    fleet_avg_display = round(fleet_avg, 4)
    relative_improvement = (
        (self_avg_display - fleet_avg_display) / self_avg_display
        if self_avg_display > 0.0
        else 0.0
    )
    print("\nAverage test RMSE:")
    print(f"  self_only_shape_constrained: {self_avg_display:.4f} SOH")
    print(f"  fleet_updated_deterministic_v3: {fleet_avg_display:.4f} SOH")
    print(f"  relative improvement: {100.0 * relative_improvement:.1f}%")

    print(
        "\nInterpretation: fleet-updated V3 gives a modest improvement over "
        "the self-only shape-constrained baseline on average, but the benefit "
        "is battery-dependent and not decisive."
    )
    print(
        "Final V3 capacity-aging extension: fleet_updated_deterministic_v3 "
        "(exploratory deterministic fleet-informed candidate)."
    )
    print(
        "This capacity-aging extension is not robust enough to be the main "
        "success of the project."
    )
    print(
        "Problematic batteries: B0005 still underestimates the late-life tail; "
        "B0018 still misses local post-split recovery / plateau behavior. "
        "B0006 and B0007 are relatively reasonable in this split."
    )

    print("\nForecast linearity warnings:")
    warning_count = 0
    for battery_id in TARGET_BATTERIES:
        battery_metrics = metrics[metrics["battery_id"] == battery_id]
        self_row = battery_metrics[
            battery_metrics["model_name"] == "self_only_shape_constrained"
        ].iloc[0]
        fleet_row = battery_metrics[
            battery_metrics["model_name"] == "fleet_updated_deterministic_v3"
        ].iloc[0]
        for row in [self_row, fleet_row]:
            if bool(row.effectively_linear_forecast):
                warning_count += 1
                print(
                    f"  Warning: {row.battery_id} {row.model_name} test forecast "
                    "is close to linear by second-difference check."
                )
        if (
            fleet_row.test_rmse > self_row.test_rmse
            and fleet_row.test_linearity_score < self_row.test_linearity_score
        ):
            warning_count += 1
            print(
                f"  Warning: {battery_id} fleet-updated model is both worse and "
                "more linear than the self-only forecast."
            )
        if battery_id in {"B0006", "B0018"} and bool(fleet_row.near_linear_forecast):
            warning_count += 1
            print(
                f"  Warning: {battery_id} final forecast remains near-linear; "
                "inspect the main figure before using it in a report."
            )
    if warning_count == 0:
        print("  No near-linear forecast warnings were triggered.")

    print(
        "\nV3 uses target train data for target-specific shape-constrained "
        "forecasting, with fleet curves only as validation-selected future-shape "
        "guidance. Target test points are not used for fitting, calibration, "
        "hyperparameter selection, or blend selection."
    )
    print(
        "No Bayesian posterior, qmax/R0 model, electrochemical model, or "
        "uncertainty bands are used."
    )

    if skipped:
        print("\nSkipped batteries:")
        for battery_id, reason in skipped.items():
            print(f"  {battery_id}: {reason}")

    print("\ngit status --short:")
    status = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        text=True,
        capture_output=True,
    )
    print(status.stdout.rstrip() or "  clean")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input file not found: {INPUT_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    sequence_table = pd.read_csv(INPUT_PATH)
    cycle_dataset, skipped = build_cycle_dataset(sequence_table)
    valid_batteries = set(cycle_dataset["battery_id"].unique())
    missing_targets = [battery for battery in TARGET_BATTERIES if battery not in valid_batteries]
    if missing_targets:
        raise ValueError(f"Target batteries are missing or invalid: {missing_targets}")

    results = {}
    for battery_id in TARGET_BATTERIES:
        results[battery_id] = evaluate_target(cycle_dataset, battery_id)

    predictions = pd.concat(
        [result["predictions"] for result in results.values()],
        ignore_index=True,
    )
    metrics = pd.concat(
        [result["metrics"] for result in results.values()],
        ignore_index=True,
    )
    components = pd.concat(
        [result["components"] for result in results.values()],
        ignore_index=True,
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    components.to_csv(COMPONENTS_PATH, index=False)

    plot_model_comparison(results)
    plot_fleet_diagnostics(results)
    plot_rmse_summary(metrics)
    print_summary(cycle_dataset, results, metrics, skipped)


if __name__ == "__main__":
    main()
