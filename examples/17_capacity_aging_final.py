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

PREDICTIONS_PATH = RESULTS_DIR / "capacity_aging_final_predictions.csv"
METRICS_PATH = RESULTS_DIR / "capacity_aging_final_metrics.csv"
FLEET_DIAGNOSTICS_FIGURE = FIGURES_DIR / "capacity_aging_final_fleet_diagnostics.png"
FORECAST_COMPARISON_FIGURE = FIGURES_DIR / "capacity_aging_final_forecast_comparison.png"
RMSE_SUMMARY_FIGURE = FIGURES_DIR / "capacity_aging_final_rmse_summary.png"
B0005_SELF_BASELINE_FIGURE = FIGURES_DIR / "capacity_aging_B0005_self_baseline.png"
REPORT_PATH = Path("capacity_aging_final_report.md")

TARGET_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
REQUIRED_COLUMNS = [
    "battery_id",
    "discharge_index",
    "temperature_c",
    "capacity_ah",
    "cycle_energy_kwh",
]

SPLIT_X = 0.70
MIN_VALID_CYCLES = 20
MAX_ALLOWED_SOH = 1.5
MAX_ALLOWED_MEDIAN_SOH = 1.2
MIN_ALLOWED_SOH = 0.3
MAX_SPLIT_UPWARD_JUMP = 0.003
PREDICTION_INCREASE_TOL = 0.002
SOH_RANGE = (0.4, 1.05)
TOP_K_FLEET = 5
LINEARITY_SECOND_DIFF_TOL = 2.5e-5

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
    """Prediction-only monotone trend; measured SOH is kept raw in outputs."""

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


def split_at_normalized_energy(table):
    table = table.sort_values("energy_frac").reset_index(drop=True)
    train = table[table["energy_frac"] <= SPLIT_X].copy()
    test = table[table["energy_frac"] > SPLIT_X].copy()
    if len(train) < 12 or len(test) < 6:
        raise ValueError("Not enough train/test points after the x=0.70 split.")
    return train, test


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
            "forecast_curvature": 0.0,
            "curved_extrapolation_used": False,
            "near_linear_forecast": True,
        }

    train_tail_count = min(max(6, int(np.ceil(0.10 * train_len))), train_len)
    train_tail_x = train_x[-train_tail_count:]
    train_tail_pred = train_pred[-train_tail_count:]
    split_slope = estimate_slope(
        x[max(0, train_len - 3) : min(len(x), train_len + 3)],
        pred[max(0, train_len - 3) : min(len(pred), train_len + 3)],
    )
    local_slopes = np.diff(test_pred) / np.maximum(np.diff(test_x), 1e-9)
    slope_change = float(local_slopes[-1] - local_slopes[0])
    second_diff = np.diff(test_pred, n=2)
    linearity_score = float(np.mean(np.abs(second_diff))) if second_diff.size else 0.0
    near_linear = (
        linearity_score < LINEARITY_SECOND_DIFF_TOL
        and abs(slope_change) < 0.06
    )
    return {
        "split_slope": float(split_slope),
        "train_tail_slope": float(estimate_slope(train_tail_x, train_tail_pred)),
        "forecast_average_slope": float(estimate_slope(test_x, test_pred)),
        "final_test_slope": float(estimate_slope(test_x[-min(8, len(test_x)) :], test_pred[-min(8, len(test_pred)) :])),
        "forecast_slope_change": slope_change,
        "forecast_curvature": linearity_score,
        "curved_extrapolation_used": not near_linear,
        "near_linear_forecast": near_linear,
    }


def select_self_hyperparameters(train_table):
    inner_fit, validation = inner_train_validation_split(train_table)
    eval_full = pd.concat([inner_fit, validation], ignore_index=True)
    x_full = eval_full["energy_frac"].to_numpy(dtype=float)
    y_val = validation["SOH"].to_numpy(dtype=float)

    candidates = []
    for params in SELF_PARAM_GRID:
        pred_full = nonlinear_self_forecast(inner_fit, x_full, params)
        pred_full = enforce_forecast_sanity(
            pred_full,
            len(inner_fit),
            "self_validation",
            str(train_table["battery_id"].iloc[0]),
        )
        validation_rmse = rmse(y_val, pred_full[len(inner_fit) :])
        shape = forecast_shape_diagnostics(x_full, pred_full, len(inner_fit))
        candidates.append(
            {
                "params": dict(params),
                "validation_rmse": float(validation_rmse),
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
    selected = sorted(
        acceptable,
        key=lambda candidate: (
            candidate["near_linear_forecast"],
            candidate["validation_rmse"],
            candidate["params"]["acceleration"],
        ),
    )[0]
    return dict(selected["params"]), float(selected["validation_rmse"])


def build_fleet_curves(cycle_dataset, target_id):
    curves = {}
    for battery_id, table in cycle_dataset.groupby("battery_id", sort=True):
        if battery_id != target_id:
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

    components = pd.DataFrame(rows).sort_values("similarity_score").reset_index(drop=True)
    selected = components.index[: min(TOP_K_FLEET, len(components))]
    selected_scores = components.loc[selected, "similarity_score"].to_numpy(dtype=float)
    positive_scores = selected_scores[selected_scores > 0]
    tau = max(float(np.median(positive_scores)) if positive_scores.size else 1.0, 1e-6)
    weights = np.exp(-selected_scores / tau)
    weights = weights / weights.sum()
    components["selected_top_k"] = False
    components["weight"] = 0.0
    components.loc[selected, "selected_top_k"] = True
    components.loc[selected, "weight"] = weights
    return components


def fleet_prior_prediction(fleet_curves, components, eval_x, train_table, weighted=True):
    selected = components[components["selected_top_k"]].copy()
    if selected.empty:
        selected = components.nsmallest(1, "similarity_score").copy()
    weights = selected["weight"].to_numpy(dtype=float)
    if not weighted:
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
    pred[future] = self_pred[future] + blend_weight * (
        fleet_curvature_residual - self_curvature_residual
    )
    pred[:train_len] = self_pred[:train_len]
    pred[anchor_index] = self_pred[anchor_index]
    return np.minimum.accumulate(pred)


def residual_update_curve(train_x, residual_train, eval_x):
    train_x = np.asarray(train_x, dtype=float)
    residual_train = np.asarray(residual_train, dtype=float)
    eval_x = np.asarray(eval_x, dtype=float)
    x_unique, residual_unique = sorted_unique_xy(train_x, residual_train)

    if len(x_unique) >= 4 and PchipInterpolator is not None:
        interpolator = PchipInterpolator(x_unique, residual_unique, extrapolate=False)
        residual_eval = np.asarray(
            interpolator(np.clip(eval_x, x_unique[0], x_unique[-1]))
        )
    else:
        residual_eval = np.interp(eval_x, x_unique, residual_unique)

    late_n = min(5, len(x_unique))
    late_slope = min(
        estimate_slope(x_unique[-late_n:], residual_unique[-late_n:]),
        0.0,
    )
    anchor_x = float(x_unique[-1])
    anchor_residual = float(residual_unique[-1])
    future = eval_x > anchor_x
    if np.any(future):
        dx = eval_x[future] - anchor_x
        span = max(anchor_x - float(x_unique[0]), 1e-6)
        damping = 1.0 / (1.0 + 1.5 * dx / span)
        residual_eval[future] = anchor_residual + late_slope * dx * damping
    return residual_eval


def fleet_residual_calibrated_prediction(fit_table, eval_x, fleet_curves):
    components = score_fleet_curves(fleet_curves, fit_table)
    prior = fleet_prior_prediction(
        fleet_curves,
        components,
        eval_x,
        fit_table,
        weighted=True,
    )
    residual_train = fit_table["SOH"].to_numpy(dtype=float) - prior[: len(fit_table)]
    pred = prior + residual_update_curve(
        fit_table["energy_frac"].to_numpy(dtype=float),
        residual_train,
        eval_x,
    )
    return pred, prior, components


def validation_candidate_predictions(candidate, fit_table, eval_full, self_params, fleet_curves, target_id):
    x_eval = eval_full["energy_frac"].to_numpy(dtype=float)
    train_len = len(fit_table)
    self_pred = nonlinear_self_forecast(fit_table, x_eval, self_params)
    self_pred = enforce_forecast_sanity(
        self_pred,
        train_len,
        "validation_self",
        target_id,
    )
    if candidate["update_family"] == "self_only_fallback":
        return self_pred

    if candidate["update_family"] == "fleet_curvature_residual":
        components = score_fleet_curves(fleet_curves, fit_table)
        weighted_prior = fleet_prior_prediction(
            fleet_curves,
            components,
            x_eval,
            fit_table,
            weighted=True,
        )
        pred = blend_self_with_fleet_curvature(
            x_eval,
            self_pred,
            weighted_prior,
            train_len,
            candidate["blend_weight"],
        )
    elif candidate["update_family"] == "fleet_prior_residual_calibration":
        pred, _, _ = fleet_residual_calibrated_prediction(
            fit_table,
            x_eval,
            fleet_curves,
        )
    else:
        raise ValueError(f"Unknown update family: {candidate['update_family']}")

    return enforce_forecast_sanity(
        pred,
        train_len,
        f"validation_{candidate['update_family']}",
        target_id,
    )


def select_final_update_family(train_table, target_id, self_params, fleet_curves):
    candidates = [{"update_family": "self_only_fallback", "blend_weight": 0.0}]
    candidates.extend(
        {
            "update_family": "fleet_curvature_residual",
            "blend_weight": blend_weight,
        }
        for blend_weight in [0.10, 0.20, 0.30]
    )
    candidates.append(
        {
            "update_family": "fleet_prior_residual_calibration",
            "blend_weight": 0.0,
        }
    )

    fold_fractions = [0.55, 0.65, 0.75]
    fold_rows = []
    for fold_fraction in fold_fractions:
        split_index = int(np.floor(fold_fraction * len(train_table)))
        split_index = min(max(split_index, 8), len(train_table) - 4)
        fit_table = train_table.iloc[:split_index].copy()
        validation = train_table.iloc[split_index:].copy()
        eval_full = pd.concat([fit_table, validation], ignore_index=True)
        y_val = validation["SOH"].to_numpy(dtype=float)

        for candidate in candidates:
            pred = validation_candidate_predictions(
                candidate,
                fit_table,
                eval_full,
                self_params,
                fleet_curves,
                target_id,
            )
            fold_rows.append(
                {
                    **candidate,
                    "fold_fraction": fold_fraction,
                    "validation_rmse": rmse(y_val, pred[len(fit_table) :]),
                }
            )

    validation = pd.DataFrame(fold_rows)
    self_by_fold = validation[
        validation["update_family"] == "self_only_fallback"
    ].set_index("fold_fraction")["validation_rmse"]

    candidate_summaries = []
    for keys, group in validation.groupby(["update_family", "blend_weight"]):
        update_family, blend_weight = keys
        group = group.copy()
        fold_delta = []
        for row in group.itertuples(index=False):
            fold_delta.append(
                float(self_by_fold.loc[row.fold_fraction] - row.validation_rmse)
            )
        mean_rmse = float(group["validation_rmse"].mean())
        mean_self = float(self_by_fold.mean())
        minimum_fold_delta = float(np.min(fold_delta))
        mean_improvement = mean_self - mean_rmse
        candidate_summaries.append(
            {
                "update_family": update_family,
                "blend_weight": float(blend_weight),
                "validation_rmse": mean_rmse,
                "self_validation_rmse": mean_self,
                "mean_validation_improvement": mean_improvement,
                "minimum_fold_improvement": minimum_fold_delta,
                "eligible": (
                    update_family == "self_only_fallback"
                    or (
                        mean_improvement > max(0.0010, 0.05 * mean_self)
                        and minimum_fold_delta >= -1e-9
                    )
                ),
            }
        )

    summary = pd.DataFrame(candidate_summaries)
    eligible = summary[summary["eligible"]].copy()
    selected = eligible.sort_values(
        ["validation_rmse", "blend_weight"],
        ascending=[True, True],
    ).iloc[0].to_dict()
    selected["validation_summary"] = summary.to_dict(orient="records")
    return selected


def limitation_note(battery_id, self_rmse, fleet_rmse, shape, final_bias, train_rmse):
    notes = []
    if fleet_rmse >= self_rmse:
        notes.append("fleet update was not accepted or did not improve versus self-only")
    if abs(final_bias) > 0.025:
        notes.append("late-life final SOH bias remains material")
    if train_rmse > 0.010:
        notes.append("first 70% target fit is noisy or imperfect")
    if shape["near_linear_forecast"]:
        notes.append("forecast is close to linear after the split")
    if battery_id in {"B0005", "B0018"}:
        notes.append("tail behavior/local recovery remains difficult")
    if not notes:
        notes.append("reasonable deterministic forecast under the clean split")
    return "; ".join(notes)


def prediction_frame(battery_id, full_table, split_labels, model_name, model_role, pred):
    table = full_table.copy()
    table["target_battery_id"] = battery_id
    table["split"] = split_labels
    table["model_name"] = model_name
    table["model_role"] = model_role
    table["predicted_soh"] = pred
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
            "model_role",
            "predicted_soh",
            "predicted_capacity_ah",
        ]
    ]


def metric_row(battery_id, model_name, model_role, train_table, test_table, pred, params, shape, self_rmse=None):
    n_train = len(train_table)
    y_train = train_table["SOH"].to_numpy(dtype=float)
    y_test = test_table["SOH"].to_numpy(dtype=float)
    train_pred = pred[:n_train]
    test_pred = pred[n_train:]
    test_rmse = rmse(y_test, test_pred)
    test_mae = mae(y_test, test_pred)
    train_rmse = rmse(y_train, train_pred)
    final_bias = float(test_pred[-1] - y_test[-1])
    improvement = (
        100.0 * (self_rmse - test_rmse) / self_rmse
        if self_rmse is not None and self_rmse > 0.0
        else 0.0
    )
    return {
        "battery_id": battery_id,
        "model_name": model_name,
        "model_role": model_role,
        "n_train": int(len(train_table)),
        "n_test": int(len(test_table)),
        "train_rmse_soh": train_rmse,
        "test_rmse_soh": test_rmse,
        "test_mae_soh": test_mae,
        "relative_improvement_vs_self_percent": improvement,
        "selected_parameters": json.dumps(params, sort_keys=True),
        "selected_update_family": params.get("update_family", "self_only_fallback"),
        "selected_blend_weight": float(params.get("blend_weight", 0.0)),
        "curved_extrapolation_used": bool(shape["curved_extrapolation_used"]),
        "near_linear_forecast": bool(shape["near_linear_forecast"]),
        "forecast_curvature": float(shape["forecast_curvature"]),
        "forecast_average_slope": float(shape["forecast_average_slope"]),
        "final_test_slope": float(shape["final_test_slope"]),
        "final_soh_bias": final_bias,
        "notes": limitation_note(battery_id, self_rmse or test_rmse, test_rmse, shape, final_bias, train_rmse),
    }


def evaluate_target(cycle_dataset, battery_id):
    target_table = cycle_dataset[cycle_dataset["battery_id"] == battery_id].copy()
    train_table, test_table = split_at_normalized_energy(target_table)
    full_table = pd.concat([train_table, test_table], ignore_index=True)
    x_full = full_table["energy_frac"].to_numpy(dtype=float)
    split_labels = np.array(["train"] * len(train_table) + ["test"] * len(test_table))

    self_params, self_validation_rmse = select_self_hyperparameters(train_table)
    self_pred = nonlinear_self_forecast(train_table, x_full, self_params)
    self_pred = enforce_forecast_sanity(
        self_pred,
        len(train_table),
        "self_only_baseline",
        battery_id,
    )
    self_shape = forecast_shape_diagnostics(x_full, self_pred, len(train_table))

    fleet_curves = build_fleet_curves(cycle_dataset, battery_id)
    fleet_selection = select_final_update_family(
        train_table,
        battery_id,
        self_params,
        fleet_curves,
    )
    components = score_fleet_curves(fleet_curves, train_table)
    weighted_prior = fleet_prior_prediction(
        fleet_curves,
        components,
        x_full,
        train_table,
        weighted=True,
    )
    uniform_prior = fleet_prior_prediction(
        fleet_curves,
        components,
        x_full,
        train_table,
        weighted=False,
    )
    if fleet_selection["update_family"] == "self_only_fallback":
        fleet_pred = self_pred.copy()
    elif fleet_selection["update_family"] == "fleet_curvature_residual":
        fleet_pred = blend_self_with_fleet_curvature(
            x_full,
            self_pred,
            weighted_prior,
            len(train_table),
            fleet_selection["blend_weight"],
        )
    elif fleet_selection["update_family"] == "fleet_prior_residual_calibration":
        fleet_pred, weighted_prior, components = fleet_residual_calibrated_prediction(
            train_table,
            x_full,
            fleet_curves,
        )
    else:
        raise ValueError(f"Unknown selected update family: {fleet_selection['update_family']}")
    fleet_pred = enforce_forecast_sanity(
        fleet_pred,
        len(train_table),
        "fleet_updated_deterministic_final",
        battery_id,
    )
    fleet_shape = forecast_shape_diagnostics(x_full, fleet_pred, len(train_table))

    self_metric = metric_row(
        battery_id,
        "self_only_baseline",
        "baseline",
        train_table,
        test_table,
        self_pred,
        {
            **self_params,
            "validation_rmse": self_validation_rmse,
            "blend_weight": 0.0,
            "update_family": "self_only_fallback",
        },
        self_shape,
    )
    fleet_metric = metric_row(
        battery_id,
        "fleet_updated_deterministic_final",
        "final_model",
        train_table,
        test_table,
        fleet_pred,
        {
            **self_params,
            "validation_rmse": fleet_selection["validation_rmse"],
            "self_validation_rmse": fleet_selection["self_validation_rmse"],
            "minimum_fold_improvement": fleet_selection["minimum_fold_improvement"],
            "mean_validation_improvement": fleet_selection["mean_validation_improvement"],
            "blend_weight": fleet_selection["blend_weight"],
            "update_family": fleet_selection["update_family"],
        },
        fleet_shape,
        self_rmse=self_metric["test_rmse_soh"],
    )

    predictions = pd.concat(
        [
            prediction_frame(
                battery_id,
                full_table,
                split_labels,
                "self_only_baseline",
                "baseline",
                self_pred,
            ),
            prediction_frame(
                battery_id,
                full_table,
                split_labels,
                "fleet_updated_deterministic_final",
                "final_model",
                fleet_pred,
            ),
            prediction_frame(
                battery_id,
                full_table,
                split_labels,
                "uniform_fleet_prior_diagnostic_only",
                "diagnostic_prior",
                uniform_prior,
            ),
            prediction_frame(
                battery_id,
                full_table,
                split_labels,
                "similarity_weighted_prior_diagnostic_only",
                "diagnostic_prior",
                weighted_prior,
            ),
        ],
        ignore_index=True,
    )

    components.insert(0, "battery_id", battery_id)
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


def plot_fleet_diagnostics(results):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, sharey=True)
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
                alpha=0.50,
                label="Individual fleet curves" if index == 0 else None,
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
        ax.plot(
            result["x_full"],
            result["uniform_prior"],
            color="#6f6fbd",
            linestyle=":",
            linewidth=2.0,
            label="Uniform fleet prior diagnostic only",
        )
        ax.plot(
            result["x_full"],
            result["weighted_prior"],
            color="#e68613",
            linestyle="--",
            linewidth=2.0,
            label="Similarity-weighted prior diagnostic only",
        )
        ax.plot(
            result["x_full"],
            result["fleet_pred"],
            color="#208744",
            linewidth=2.8,
            label="Final target-specific forecast",
        )
        ax.axvline(SPLIT_X, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(battery_id)
        ax.set_xlabel("Normalized cumulative energy age")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Capacity-aging final fleet diagnostics: priors are diagnostic, final forecast is target-specific",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    fig.savefig(FLEET_DIAGNOSTICS_FIGURE, dpi=200)
    plt.close(fig)


def plot_forecast_comparison(results):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, battery_id in zip(axes, TARGET_BATTERIES):
        result = results[battery_id]
        full = result["full"]
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
            SPLIT_X,
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
    fig.suptitle("Capacity-aging final forecast comparison", fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(FORECAST_COMPARISON_FIGURE, dpi=200)
    plt.close(fig)

def plot_b0005_self_baseline(results):
    """Plot B0005 measured SOH, self-only baseline, train/test split, and 80% threshold."""
    target_id = "B0005"

    if target_id not in results:
        print("Skipped B0005 self-only baseline plot: B0005 not found in results.")
        return

    result = results[target_id]

    full = result["full"]
    x_full = result["x_full"]
    self_pred = result["self_pred"]

    metric_row = result["metrics"][
        result["metrics"]["model_name"] == "self_only_baseline"
    ]

    rmse_text = ""
    if not metric_row.empty:
        b0005_rmse = float(metric_row["test_rmse_soh"].iloc[0])
        rmse_text = f", test RMSE {b0005_rmse:.4f} SOH"

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        full["energy_frac"],
        full["SOH"],
        "o-",
        markersize=3,
        linewidth=1.1,
        label="Measured raw SOH",
    )

    ax.plot(
        x_full,
        self_pred,
        linewidth=2.4,
        label="Self-only baseline",
    )

    ax.axvline(
        SPLIT_X,
        linestyle=":",
        linewidth=1.8,
        label="Train/test split",
    )

    ax.axhline(
        0.80,
        linestyle="--",
        linewidth=1.8,
        label="80% initial capacity",
    )

    ax.set_title(f"B0005 self-only capacity-aging baseline{rmse_text}")
    ax.set_xlabel("Normalized cumulative energy age")
    ax.set_ylabel("SOH")
    ax.set_ylim(0.55, 1.05)
    ax.grid(alpha=0.30)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(B0005_SELF_BASELINE_FIGURE, dpi=200)
    plt.close(fig)

def plot_rmse_summary(metrics):
    final_metrics = metrics[
        metrics["model_name"].isin(
            ["self_only_baseline", "fleet_updated_deterministic_final"]
        )
    ].copy()
    summary = final_metrics.groupby("model_name", as_index=False)["test_rmse_soh"].mean()
    order = ["self_only_baseline", "fleet_updated_deterministic_final"]
    labels = {
        "self_only_baseline": "Self-only\nbaseline",
        "fleet_updated_deterministic_final": "Fleet-informed\nfinal",
    }
    colors = {
        "self_only_baseline": "#2f6db3",
        "fleet_updated_deterministic_final": "#208744",
    }
    summary["order"] = summary["model_name"].map({name: index for index, name in enumerate(order)})
    summary = summary.sort_values("order")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(
        np.arange(len(summary)),
        summary["test_rmse_soh"],
        color=[colors[name] for name in summary["model_name"]],
    )
    ax.set_xticks(
        np.arange(len(summary)),
        [labels[name] for name in summary["model_name"]],
    )
    ax.set_ylabel("Mean test RMSE (SOH)")
    ax.set_title("Capacity-aging final RMSE summary")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary["test_rmse_soh"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(RMSE_SUMMARY_FIGURE, dpi=200)
    plt.close(fig)


def write_report(cycle_dataset, results, metrics, skipped):
    final_metrics = metrics[metrics["model_name"] == "fleet_updated_deterministic_final"].copy()
    self_metrics = metrics[metrics["model_name"] == "self_only_baseline"].copy()
    mean_self = float(self_metrics["test_rmse_soh"].mean())
    mean_final = float(final_metrics["test_rmse_soh"].mean())
    mean_improvement = 100.0 * (mean_self - mean_final) / mean_self
    improved_text = "improves" if mean_final < mean_self else "does not improve"

    report = """# Capacity-Aging Final Example 17

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

"""
    report += (
        f"The final fleet-informed model {improved_text} over the self-only baseline "
        f"on average test RMSE: self-only = {mean_self:.5f}, final = {mean_final:.5f}, "
        f"relative change = {mean_improvement:.2f}%. The gain is modest and battery-dependent.\n\n"
    )
    report += "| battery | self RMSE | final RMSE | improvement | note |\n"
    report += "| --- | ---: | ---: | ---: | --- |\n"
    pivot = metrics.pivot(index="battery_id", columns="model_name", values="test_rmse_soh")
    notes = final_metrics.set_index("battery_id")["notes"]
    improvements = final_metrics.set_index("battery_id")["relative_improvement_vs_self_percent"]
    for battery_id in TARGET_BATTERIES:
        family = final_metrics.set_index("battery_id").loc[
            battery_id,
            "selected_update_family",
        ]
        report += (
            f"| {battery_id} | {pivot.loc[battery_id, 'self_only_baseline']:.5f} | "
            f"{pivot.loc[battery_id, 'fleet_updated_deterministic_final']:.5f} | "
            f"{improvements.loc[battery_id]:.2f}% | {family}: {notes.loc[battery_id]} |\n"
        )

    report += """
## 9. Per-battery findings

B0005 and B0018 remain difficult late-life targets. Their tails and local recovery/plateau behavior are not fully represented by the monotone low-complexity forecast. B0006 can show imperfect first-70% fit while still producing a comparatively good future forecast; this should be interpreted as battery-dependent behavior, not a general success claim. B0007 is the clearest case where fleet curvature can help under this split.

## 10. Limitations

The fleet-informed model gives only modest and battery-dependent improvement. It is not a full Bayesian HBPINN reproduction, not a vMLP posterior method, and not a qmax/R0 posterior update. It is still useful because it completes the project pipeline from voltage response to cycle-level aging evaluation in a reproducible way, with explicit train/test separation and clear diagnostic figures.
"""
    if skipped:
        report += "\nSkipped non-target batteries during cleaning:\n\n"
        for battery_id, reason in skipped.items():
            report += f"- {battery_id}: {reason}\n"

    REPORT_PATH.write_text(report, encoding="utf-8")


def print_summary(metrics):
    generated = [
        Path("examples/17_capacity_aging_final.py"),
        PREDICTIONS_PATH,
        METRICS_PATH,
        FLEET_DIAGNOSTICS_FIGURE,
        FORECAST_COMPARISON_FIGURE,
        B0005_SELF_BASELINE_FIGURE,
        RMSE_SUMMARY_FIGURE,
        REPORT_PATH,
    ]
    print("\nGenerated files:")
    for path in generated:
        print(f"  {path}")

    print("\nKey RMSE table:")
    key = metrics[
        metrics["model_name"].isin(
            ["self_only_baseline", "fleet_updated_deterministic_final"]
        )
    ][
        [
            "battery_id",
            "model_name",
            "test_rmse_soh",
            "test_mae_soh",
            "relative_improvement_vs_self_percent",
            "notes",
        ]
    ]
    print(key.to_string(index=False))

    mean_rmse = metrics.groupby("model_name")["test_rmse_soh"].mean()
    self_rmse = float(mean_rmse["self_only_baseline"])
    final_rmse = float(mean_rmse["fleet_updated_deterministic_final"])
    change = 100.0 * (self_rmse - final_rmse) / self_rmse
    print("\nAverage RMSE:")
    print(f"  self_only_baseline: {self_rmse:.5f}")
    print(f"  fleet_updated_deterministic_final: {final_rmse:.5f}")
    print(f"  relative improvement: {change:.2f}%")
    if final_rmse < self_rmse:
        print("  Final fleet-informed model improves over self-only on average.")
    else:
        print("  Final fleet-informed model does not improve over self-only on average.")

    print("\nLimitations:")
    final_rows = metrics[metrics["model_name"] == "fleet_updated_deterministic_final"]
    for row in final_rows.itertuples(index=False):
        print(f"  {row.battery_id}: {row.notes}")

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

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)

    plot_fleet_diagnostics(results)
    plot_forecast_comparison(results)
    plot_b0005_self_baseline(results)
    plot_rmse_summary(metrics)
    write_report(cycle_dataset, results, metrics, skipped)
    print_summary(metrics)


if __name__ == "__main__":
    main()
