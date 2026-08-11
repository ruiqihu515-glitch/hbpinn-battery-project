"""Reusable deterministic capacity-aging computations."""

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .metrics import mae, rmse

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


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
    """Return the first non-missing value, or ``NaN`` when none exists."""
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def robust_capacity(series):
    """Return the median finite positive capacity, or ``NaN`` if unavailable."""
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.empty:
        return np.nan
    return float(values.median())


def sorted_unique_xy(x, y):
    """Return finite observations sorted by x, averaging y at duplicate x values."""
    table = (
        pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("x")
    )
    grouped = table.groupby("x", as_index=False)["y"].mean()
    return grouped["x"].to_numpy(dtype=float), grouped["y"].to_numpy(dtype=float)


def estimate_slope(x, y):
    """Estimate a linear slope, returning zero when x is insufficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


class MonotonePchipCurve:
    """Prediction-only monotone trend; measured SOH is kept raw in outputs."""

    def __init__(self, x, y):
        """Fit a non-increasing interpolation curve to unique finite samples."""
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
        """Evaluate the curve with clipped, non-increasing extrapolation."""
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
    """Convert time-step discharge sequences into a cycle-level aging dataset.

    The resulting table contains robust cycle capacity, robust initial
    capacity, ``SOH = capacity_ah / initial_capacity_ah``, cycle and cumulative
    energy aging variables, and retained temperature summary statistics.
    Temperature is retained for analysis, not used as a capacity-prediction
    feature. Normalized energy uses each battery's observed full-life maximum
    cumulative energy, so its definition is retrospective.

    Returns the valid cycle table and a mapping of skipped batteries to their
    validation reasons.
    """
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
    """Split a battery table at the 0.70 normalized-energy threshold."""
    table = table.sort_values("energy_frac").reset_index(drop=True)
    train = table[table["energy_frac"] <= SPLIT_X].copy()
    test = table[table["energy_frac"] > SPLIT_X].copy()
    if len(train) < 12 or len(test) < 6:
        raise ValueError("Not enough train/test points after the x=0.70 split.")
    return train, test


def inner_train_validation_split(train_table):
    """Partition available training cycles into chronological fit and validation sets."""
    split_index = int(np.floor(0.72 * len(train_table)))
    split_index = min(max(split_index, 5), len(train_table) - 2)
    return train_table.iloc[:split_index].copy(), train_table.iloc[split_index:].copy()


def nonlinear_self_forecast(fit_table, eval_x, params):
    """Create a nonlinear self-only SOH forecast for the target battery.

    A monotone fitted trend supplies the observed-range prediction and anchor.
    Late fitted degradation rates, their change, and the selected acceleration
    and power parameters construct the curved extrapolation beyond that anchor.
    """
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
    """Apply finite, monotonic, split-continuity, and SOH-range forecast checks."""
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
    """Summarize forecast slopes, curvature, and near-linearity after the split."""
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
    """Select self-forecast parameters by an inner fit/validation procedure.

    Selection uses only the available training table; the final test set is not
    involved.
    """
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
    """Build reusable monotone degradation curves for non-target fleet batteries."""
    curves = {}
    for battery_id, table in cycle_dataset.groupby("battery_id", sort=True):
        if battery_id != target_id:
            curves[battery_id] = MonotonePchipCurve(table["energy_frac"], table["SOH"])
    return curves


def score_fleet_curves(fleet_curves, train_table):
    """Rank fleet curves against the target's observed training behavior.

    Each curve is anchor-shifted to the target and scored by overall training
    RMSE, late-training RMSE, and half the late-slope mismatch. The best curves
    receive normalized similarity weights.
    """
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
    """Construct a weighted or uniform prior from selected, anchor-shifted fleet curves."""
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
    """Blend fleet curvature residuals into the self forecast beyond training."""
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
    """Interpolate training residuals and damp their late-slope extrapolation."""
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
    """Combine a fleet prior with a target-specific residual correction.

    The correction is calibrated from the observed target training data and is
    added to the selected similarity-weighted fleet prior.
    """
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
    """Generate one update-family candidate for a training validation fold."""
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
    """Apply a train-only validation gate to choose the final update family.

    ``self_only_fallback`` is always available. The fleet candidates are
    ``fleet_curvature_residual``, evaluated at curvature blend strengths 0.10,
    0.20, and 0.30, and ``fleet_prior_residual_calibration``. Fleet candidates
    are eligible only when they meet the implemented mean-improvement and
    per-fold validation criteria, so fleet information is not forced. The
    eligible family is selected by validation behavior, never test performance.
    """
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
    """Build a concise diagnostic note from model errors and forecast shape."""
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
    """Package cycle observations and one model's predictions into a table."""
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
    """Compute one model's train/test metrics and diagnostic metadata."""
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
    """Run the high-level cycle-level evaluation for one target battery.

    The workflow applies the 0.70 normalized-energy train/test split, whose
    denominator is the observed full-life cumulative-energy maximum, fits the
    self-only forecast, builds and scores non-target fleet curves, and uses the
    train-only validation gate to select an update family. It then generates
    predictions and returns the existing metrics, fleet diagnostics, shape
    diagnostics, split tables, and selection details.

    The final forecast is validation-selected and may remain self-only; a fleet
    update is not applied automatically.
    """
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
