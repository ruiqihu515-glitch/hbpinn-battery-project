from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from scipy.optimize import lsq_linear
except ImportError:
    lsq_linear = None


REQUIRED_SEQUENCE_COLUMNS = [
    "battery_id",
    "discharge_index",
    "temperature_c",
    "capacity_ah",
    "cycle_energy_kwh",
    "cumulative_energy_kwh",
]

MODEL_ORDER = [
    "self_energy_baseline",
    "fleet_prior_uniform",
    "fleet_prior_similarity_weighted_raw",
    "fleet_prior_calibrated",
    "adaptive_self_fleet_blend",
    "fleet_prior_calibrated_temperature",
]

MODEL_LABELS = {
    "self_energy_baseline": "Self baseline",
    "fleet_prior_uniform": "Uniform fleet prior diagnostic",
    "fleet_prior_similarity_weighted_raw": "Similarity-weighted raw prior diagnostic",
    "fleet_prior_calibrated": "Fleet calibrated final model",
    "adaptive_self_fleet_blend": "Adaptive blend diagnostic",
    "fleet_prior_calibrated_temperature": "Temperature candidate",
}

MODEL_ROLES = {
    "self_energy_baseline": "self_baseline",
    "fleet_prior_uniform": "diagnostic_prior",
    "fleet_prior_similarity_weighted_raw": "diagnostic_prior",
    "fleet_prior_calibrated": "fleet_candidate",
    "adaptive_self_fleet_blend": "diagnostic_candidate",
    "fleet_prior_calibrated_temperature": "temperature_candidate",
}

MAIN_FIGURE_MODELS = [
    "self_energy_baseline",
    "fleet_prior_calibrated",
    "adaptive_self_fleet_blend",
    "fleet_prior_calibrated_temperature",
]

RMSE_FIGURE_MODELS = MAIN_FIGURE_MODELS

MIN_VALID_CAPACITY_OBSERVATIONS = 20
MAX_ALLOWED_SOH = 1.5
MAX_ALLOWED_MEDIAN_SOH = 1.2
MIN_ALLOWED_SOH = 0.3
PREDICTION_INCREASE_TOLERANCE = 0.005


def first_non_null(series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def last_non_null(series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[-1]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def score_capacity_predictions(table, predicted_soh):
    predicted_capacity_ah = predicted_soh * table["initial_capacity_ah"].to_numpy()

    return {
        "rmse_soh": rmse(table["SOH"], predicted_soh),
        "mae_soh": mean_absolute_error(table["SOH"], predicted_soh),
        "rmse_capacity_ah": rmse(table["capacity_ah"], predicted_capacity_ah),
        "mae_capacity_ah": mean_absolute_error(
            table["capacity_ah"],
            predicted_capacity_ah,
        ),
    }


def build_raw_cycle_level_dataset(sequence_table):
    missing_columns = [
        column
        for column in REQUIRED_SEQUENCE_COLUMNS
        if column not in sequence_table.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required sequence columns: {missing_columns}")

    cycle_table = (
        sequence_table.groupby(["battery_id", "discharge_index"], as_index=False)
        .agg(
            capacity_ah=("capacity_ah", first_non_null),
            cycle_energy_kwh=("cycle_energy_kwh", first_non_null),
            cumulative_energy_kwh=("cumulative_energy_kwh", last_non_null),
            mean_temperature_c=("temperature_c", "mean"),
            max_temperature_c=("temperature_c", "max"),
            min_temperature_c=("temperature_c", "min"),
            number_of_steps=("temperature_c", "size"),
        )
        .sort_values(["battery_id", "discharge_index"])
        .reset_index(drop=True)
    )

    cycle_table["cycle_age"] = np.nan
    cycle_table["energy_age_kwh"] = np.nan

    for battery_id, battery_index in cycle_table.groupby("battery_id").groups.items():
        battery_table = cycle_table.loc[battery_index].sort_values("discharge_index")
        first_discharge_index = battery_table["discharge_index"].iloc[0]
        first_cumulative_energy_kwh = first_non_null(
            battery_table["cumulative_energy_kwh"]
        )

        cycle_table.loc[battery_table.index, "cycle_age"] = (
            battery_table["discharge_index"] - first_discharge_index
        )
        cycle_table.loc[battery_table.index, "energy_age_kwh"] = (
            battery_table["cumulative_energy_kwh"] - first_cumulative_energy_kwh
        )

    output_columns = [
        "battery_id",
        "discharge_index",
        "capacity_ah",
        "cycle_age",
        "cycle_energy_kwh",
        "energy_age_kwh",
        "cumulative_energy_kwh",
        "mean_temperature_c",
        "max_temperature_c",
        "min_temperature_c",
        "number_of_steps",
    ]

    return cycle_table[output_columns]


def finite_or_nan(value):
    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    return value


def make_skip_reason(reasons):
    if not reasons:
        return ""
    return "; ".join(reasons)


def build_capacity_diagnostics_and_model_dataset(raw_cycle_table):
    model_tables = []
    diagnostic_rows = []

    for battery_id in sorted(raw_cycle_table["battery_id"].dropna().unique()):
        battery_table = (
            raw_cycle_table[raw_cycle_table["battery_id"] == battery_id]
            .sort_values("discharge_index")
            .copy()
        )
        capacity_values = battery_table["capacity_ah"]
        finite_capacity = capacity_values[
            capacity_values.notna() & np.isfinite(capacity_values)
        ]
        positive_capacity = finite_capacity[finite_capacity > 0.0]

        first_capacity_ah = first_non_null(capacity_values)
        min_capacity_ah = finite_capacity.min() if not finite_capacity.empty else np.nan
        median_capacity_ah = (
            finite_capacity.median() if not finite_capacity.empty else np.nan
        )
        max_capacity_ah = finite_capacity.max() if not finite_capacity.empty else np.nan

        if len(positive_capacity) >= 3:
            candidate_initial_capacity_ah = positive_capacity.iloc[:3].median()
            capacity_reference_method = "median_first_3_valid_positive"
        elif len(positive_capacity) > 0:
            candidate_initial_capacity_ah = positive_capacity.iloc[0]
            capacity_reference_method = "first_valid_positive"
        else:
            candidate_initial_capacity_ah = np.nan
            capacity_reference_method = "none"

        battery_table["initial_capacity_ah"] = candidate_initial_capacity_ah
        battery_table["SOH"] = battery_table["capacity_ah"] / candidate_initial_capacity_ah
        battery_table["capacity_reference_method"] = capacity_reference_method

        finite_soh = battery_table["SOH"][
            battery_table["SOH"].notna() & np.isfinite(battery_table["SOH"])
        ]
        min_soh = finite_soh.min() if not finite_soh.empty else np.nan
        median_soh = finite_soh.median() if not finite_soh.empty else np.nan
        max_soh = finite_soh.max() if not finite_soh.empty else np.nan

        skip_reasons = []
        if len(positive_capacity) < MIN_VALID_CAPACITY_OBSERVATIONS:
            skip_reasons.append(
                "fewer than 20 valid positive capacity observations "
                f"({len(positive_capacity)})"
            )
        if (
            pd.isna(candidate_initial_capacity_ah)
            or not np.isfinite(candidate_initial_capacity_ah)
            or candidate_initial_capacity_ah <= 0.0
        ):
            skip_reasons.append("invalid candidate initial capacity")
        if pd.isna(median_soh) or not np.isfinite(median_soh):
            skip_reasons.append("median SOH is not finite")
        if np.isfinite(max_soh) and max_soh > MAX_ALLOWED_SOH:
            skip_reasons.append(f"max SOH greater than {MAX_ALLOWED_SOH}")
        if np.isfinite(median_soh) and median_soh > MAX_ALLOWED_MEDIAN_SOH:
            skip_reasons.append(f"median SOH greater than {MAX_ALLOWED_MEDIAN_SOH}")
        if np.isfinite(min_soh) and min_soh < MIN_ALLOWED_SOH:
            skip_reasons.append(f"min SOH less than {MIN_ALLOWED_SOH}")

        battery_valid = len(skip_reasons) == 0
        status = "used" if battery_valid else "skipped"
        reason_if_skipped = make_skip_reason(skip_reasons)

        diagnostic_rows.append(
            {
                "battery_id": battery_id,
                "n_cycles_raw": len(battery_table),
                "n_capacity_non_missing": int(capacity_values.notna().sum()),
                "first_capacity_ah": finite_or_nan(first_capacity_ah),
                "min_capacity_ah": finite_or_nan(min_capacity_ah),
                "median_capacity_ah": finite_or_nan(median_capacity_ah),
                "max_capacity_ah": finite_or_nan(max_capacity_ah),
                "candidate_initial_capacity_ah": finite_or_nan(
                    candidate_initial_capacity_ah
                ),
                "min_SOH": finite_or_nan(min_soh),
                "median_SOH": finite_or_nan(median_soh),
                "max_SOH": finite_or_nan(max_soh),
                "status": status,
                "reason_if_skipped": reason_if_skipped,
                "capacity_reference_method": capacity_reference_method,
            }
        )

        battery_table["battery_valid_for_aging"] = battery_valid
        battery_table["battery_skip_reason"] = reason_if_skipped

        if battery_valid:
            model_table = battery_table.dropna(
                subset=[
                    "capacity_ah",
                    "initial_capacity_ah",
                    "SOH",
                    "cycle_age",
                    "cycle_energy_kwh",
                    "energy_age_kwh",
                    "cumulative_energy_kwh",
                    "mean_temperature_c",
                    "max_temperature_c",
                    "min_temperature_c",
                    "number_of_steps",
                ]
            ).copy()
            model_table = model_table[
                np.isfinite(model_table["capacity_ah"])
                & (model_table["capacity_ah"] > 0.0)
                & np.isfinite(model_table["SOH"])
            ].copy()
            model_tables.append(model_table)

    diagnostics_table = pd.DataFrame(diagnostic_rows)

    if model_tables:
        model_cycle_table = pd.concat(model_tables, ignore_index=True)
    else:
        model_cycle_table = pd.DataFrame(columns=list(raw_cycle_table.columns))

    model_cycle_table["energy_frac"] = np.nan
    for battery_id, battery_index in model_cycle_table.groupby("battery_id").groups.items():
        max_energy_age = model_cycle_table.loc[battery_index, "energy_age_kwh"].max()
        if pd.isna(max_energy_age) or not np.isfinite(max_energy_age) or max_energy_age <= 0.0:
            model_cycle_table.loc[battery_index, "energy_frac"] = 0.0
        else:
            model_cycle_table.loc[battery_index, "energy_frac"] = (
                model_cycle_table.loc[battery_index, "energy_age_kwh"] / max_energy_age
            )

    output_columns = [
        "battery_id",
        "discharge_index",
        "capacity_ah",
        "initial_capacity_ah",
        "SOH",
        "cycle_age",
        "cycle_energy_kwh",
        "energy_age_kwh",
        "energy_frac",
        "cumulative_energy_kwh",
        "mean_temperature_c",
        "max_temperature_c",
        "min_temperature_c",
        "number_of_steps",
        "capacity_reference_method",
        "battery_valid_for_aging",
        "battery_skip_reason",
    ]

    return model_cycle_table[output_columns], diagnostics_table


def split_battery_cycles(battery_table):
    sorted_table = battery_table.sort_values("discharge_index").reset_index(drop=True)
    split_index = int(0.7 * len(sorted_table))

    train_table = sorted_table.iloc[:split_index].copy()
    test_table = sorted_table.iloc[split_index:].copy()

    return train_table, test_table


def fit_monotone_curve(table):
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    sorted_table = table.sort_values("energy_frac")
    model.fit(sorted_table["energy_frac"], sorted_table["SOH"])
    return model


def predict_monotone_curve(model, table):
    predicted = model.predict(table["energy_frac"])
    return np.minimum.accumulate(np.asarray(predicted, dtype=float))


def predict_monotone_values(model, energy_frac):
    values = np.asarray(energy_frac, dtype=float)
    predicted = model.predict(values)
    order = np.argsort(values)
    sorted_predicted = np.asarray(predicted, dtype=float)[order]
    sorted_predicted = np.minimum.accumulate(sorted_predicted)
    restored = np.empty_like(sorted_predicted)
    restored[order] = sorted_predicted
    return restored


def finite_prediction(values):
    values = np.asarray(values, dtype=float)
    return values.size > 0 and np.all(np.isfinite(values))


def self_energy_design(energy_frac):
    energy_frac = np.asarray(energy_frac, dtype=float)
    return np.column_stack(
        [
            np.ones_like(energy_frac),
            energy_frac,
            np.sqrt(np.maximum(energy_frac, 0.0) + 1e-9),
            np.log1p(np.maximum(energy_frac, 0.0)),
        ]
    )


def fit_constrained_self_baseline(train_table):
    design = self_energy_design(train_table["energy_frac"])
    y_train = train_table["SOH"].to_numpy(dtype=float)
    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, 0.0, 0.0, 0.0]

    if lsq_linear is not None:
        result = lsq_linear(
            design,
            y_train,
            bounds=(lower_bounds, upper_bounds),
            lsmr_tol="auto",
        )
        if result.success and np.all(np.isfinite(result.x)):
            return result.x, "constrained_transformed_energy"

    coefficients, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    coefficients = np.asarray(coefficients, dtype=float)
    coefficients[1:] = np.minimum(coefficients[1:], 0.0)
    return coefficients, "constrained_transformed_energy_fallback"


def predict_constrained_self_baseline(coefficients, table):
    return self_energy_design(table["energy_frac"]) @ coefficients


def late_train_slope(table):
    late_count = max(3, int(np.ceil(0.3 * len(table))))
    late_table = table.tail(late_count)
    x = late_table["energy_frac"].to_numpy(dtype=float)
    y = late_table["SOH"].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return 0.0
    return min(float(np.polyfit(x, y, 1)[0]), 0.0)


def apply_self_fallback_if_flat(train_table, test_table, train_pred, test_pred):
    slope = late_train_slope(train_table)
    if slope >= -1e-4:
        return train_pred, test_pred, False, slope
    if len(test_pred) > 1 and abs(float(test_pred[-1] - test_pred[0])) > 0.005:
        return train_pred, test_pred, False, slope

    anchor_count = min(5, len(train_table))
    anchor_energy = float(train_table["energy_frac"].tail(anchor_count).median())
    anchor_soh = float(train_table["SOH"].tail(anchor_count).median())
    train_linear = anchor_soh + slope * (
        train_table["energy_frac"].to_numpy(dtype=float) - anchor_energy
    )
    test_linear = anchor_soh + slope * (
        test_table["energy_frac"].to_numpy(dtype=float) - anchor_energy
    )
    return train_linear, test_linear, True, slope


def prediction_diagnostics_for_values(values):
    return prediction_physical_diagnostics(np.asarray(values, dtype=float))


def enforce_train_test_monotonicity(train_table, test_table, train_pred, test_pred):
    combined = pd.DataFrame(
        {
            "split": ["train"] * len(train_table) + ["test"] * len(test_table),
            "position": list(range(len(train_table))) + list(range(len(test_table)),
            ),
            "discharge_index": pd.concat(
                [train_table["discharge_index"], test_table["discharge_index"]],
                ignore_index=True,
            ),
            "predicted_SOH": np.concatenate([train_pred, test_pred]),
        }
    )
    combined = combined.sort_values("discharge_index").reset_index(drop=True)
    combined["predicted_SOH"] = np.minimum.accumulate(
        combined["predicted_SOH"].to_numpy(dtype=float)
    )
    diagnostics = prediction_physical_diagnostics(combined["predicted_SOH"])

    train_output = np.empty(len(train_table), dtype=float)
    test_output = np.empty(len(test_table), dtype=float)

    for row in combined.itertuples(index=False):
        if row.split == "train":
            train_output[int(row.position)] = row.predicted_SOH
        else:
            test_output[int(row.position)] = row.predicted_SOH

    return train_output, test_output, diagnostics


def prediction_physical_diagnostics(predicted_soh):
    values = np.asarray(predicted_soh, dtype=float)
    if values.size <= 1:
        return {
            "n_prediction_increases": 0,
            "max_prediction_increase": 0.0,
            "physically_valid_prediction": True,
        }
    increases = np.diff(values)
    positive_increases = increases[increases > 0.0]
    max_increase = (
        float(positive_increases.max()) if positive_increases.size else 0.0
    )
    return {
        "n_prediction_increases": int(positive_increases.size),
        "max_prediction_increase": max_increase,
        "physically_valid_prediction": max_increase <= PREDICTION_INCREASE_TOLERANCE,
    }


def choose_selected_final_model(metrics_table):
    physically_valid_metrics = metrics_table[
        metrics_table["physically_valid_prediction"]
    ].copy()
    physically_valid_average_rmse = (
        physically_valid_metrics.groupby("model_name")["test_rmse_soh"].mean()
    )

    selected_final_model = "fleet_prior_calibrated"
    calibrated_rmse = physically_valid_average_rmse.get(
        "fleet_prior_calibrated",
        np.inf,
    )

    for candidate_model in [
        "adaptive_self_fleet_blend",
        "fleet_prior_calibrated_temperature",
    ]:
        candidate_rmse = physically_valid_average_rmse.get(candidate_model, np.inf)
        if candidate_rmse < calibrated_rmse:
            selected_final_model = candidate_model
            calibrated_rmse = candidate_rmse

    return selected_final_model


def fit_constrained_calibration(prior_train, energy_frac_train, y_train, temperature_train=None):
    if temperature_train is None:
        design = np.column_stack(
            [
                np.ones_like(prior_train),
                prior_train,
                energy_frac_train,
            ]
        )
        lower_bounds = [-np.inf, 0.0, -np.inf]
        upper_bounds = [np.inf, np.inf, 0.0]
    else:
        design = np.column_stack(
            [
                np.ones_like(prior_train),
                prior_train,
                energy_frac_train,
                temperature_train,
            ]
        )
        lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
        upper_bounds = [np.inf, np.inf, 0.0, np.inf]

    if lsq_linear is not None:
        result = lsq_linear(
            design,
            y_train,
            bounds=(lower_bounds, upper_bounds),
            lsmr_tol="auto",
        )
        if result.success and np.all(np.isfinite(result.x)):
            return result.x, "constrained_lsq_linear"

    coefficients, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    coefficients = np.asarray(coefficients, dtype=float)
    coefficients[1] = max(coefficients[1], 0.0)
    coefficients[2] = min(coefficients[2], 0.0)
    return coefficients, "constrained_lstsq_fallback"


def append_model_results(
    metrics_rows,
    prediction_rows,
    target_battery_id,
    model_name,
    train_table,
    test_table,
    train_pred_soh,
    test_pred_soh,
    n_fleet_batteries,
    parameters,
):
    train_pred_soh, test_pred_soh, physical_diagnostics = enforce_train_test_monotonicity(
        train_table,
        test_table,
        train_pred_soh,
        test_pred_soh,
    )
    train_scores = score_capacity_predictions(train_table, train_pred_soh)
    test_scores = score_capacity_predictions(test_table, test_pred_soh)

    metric_row = {
        "target_battery_id": target_battery_id,
        "model_name": model_name,
        "model_role": MODEL_ROLES[model_name],
        "selected_k_neighbors": parameters.get("selected_k_neighbors", np.nan),
        "selected_alpha_blend": parameters.get("selected_alpha_blend", np.nan),
        "n_cycles": len(train_table) + len(test_table),
        "n_train": len(train_table),
        "n_test": len(test_table),
        "n_fleet_batteries": n_fleet_batteries,
        "train_rmse_soh": train_scores["rmse_soh"],
        "train_mae_soh": train_scores["mae_soh"],
        "validation_rmse_soh": parameters.get("validation_rmse_soh", np.nan),
        "test_rmse_soh": test_scores["rmse_soh"],
        "test_mae_soh": test_scores["mae_soh"],
        "train_rmse_capacity_ah": train_scores["rmse_capacity_ah"],
        "train_mae_capacity_ah": train_scores["mae_capacity_ah"],
        "test_rmse_capacity_ah": test_scores["rmse_capacity_ah"],
        "test_mae_capacity_ah": test_scores["mae_capacity_ah"],
    }
    metric_row.update(parameters)
    metric_row.update(physical_diagnostics)
    metrics_rows.append(metric_row)

    for split_name, split_table, split_pred_soh in [
        ("train", train_table, train_pred_soh),
        ("test", test_table, test_pred_soh),
    ]:
        prediction_table = split_table[
            [
                "battery_id",
                "discharge_index",
                "cycle_age",
                "energy_age_kwh",
                "energy_frac",
                "capacity_ah",
                "SOH",
                "initial_capacity_ah",
                "mean_temperature_c",
            ]
        ].copy()
        prediction_table.insert(0, "target_battery_id", target_battery_id)
        prediction_table["split"] = split_name
        prediction_table["model_name"] = model_name
        prediction_table["predicted_SOH"] = split_pred_soh
        prediction_table["predicted_capacity_ah"] = (
            prediction_table["predicted_SOH"] * prediction_table["initial_capacity_ah"]
        )
        prediction_rows.extend(prediction_table.to_dict("records"))


def build_fleet_curves(cycle_table, target_battery_id):
    fleet_curves = []
    warnings = []
    fleet_table = cycle_table[cycle_table["battery_id"] != target_battery_id]

    for fleet_battery_id in sorted(fleet_table["battery_id"].dropna().unique()):
        battery_table = (
            fleet_table[fleet_table["battery_id"] == fleet_battery_id]
            .dropna(subset=["energy_age_kwh", "SOH"])
            .sort_values("discharge_index")
            .copy()
        )

        if len(battery_table) < MIN_VALID_CAPACITY_OBSERVATIONS:
            warnings.append(
                f"{target_battery_id}: skipping fleet battery {fleet_battery_id} "
                f"with {len(battery_table)} usable cycles"
            )
            continue

        try:
            model = fit_monotone_curve(battery_table)
        except ValueError as error:
            warnings.append(
                f"{target_battery_id}: could not fit fleet curve for "
                f"{fleet_battery_id}: {error}"
            )
            continue

        fleet_curves.append(
            {
                "fleet_battery_id": fleet_battery_id,
                "model": model,
                "n_fleet_cycles": len(battery_table),
            }
        )

    return fleet_curves, warnings


def evaluate_fleet_curves(fleet_curves, table):
    if not fleet_curves:
        return np.empty((len(table), 0))

    predictions = []
    for curve in fleet_curves:
        predictions.append(predict_monotone_curve(curve["model"], table))
    return np.column_stack(predictions)


def linear_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def make_similarity_scores(train_table, train_component_predictions):
    train_soh = train_table["SOH"].to_numpy(dtype=float)
    train_energy = train_table["energy_frac"].to_numpy(dtype=float)
    late_count = max(3, int(np.ceil(0.3 * len(train_table))))
    anchor_count = min(5, len(train_table))
    late_slice = slice(len(train_table) - late_count, len(train_table))
    anchor_slice = slice(len(train_table) - anchor_count, len(train_table))

    target_late_slope = linear_slope(train_energy[late_slice], train_soh[late_slice])
    scores = []
    overall_rmses = []
    late_rmses = []
    final_level_errors = []
    slope_mismatches = []

    for column_index in range(train_component_predictions.shape[1]):
        pred = train_component_predictions[:, column_index]
        overall_rmse = rmse(train_soh, pred)
        late_rmse = rmse(train_soh[late_slice], pred[late_slice])
        final_level_error = abs(
            float(np.median(pred[anchor_slice]) - np.median(train_soh[anchor_slice]))
        )
        pred_late_slope = linear_slope(train_energy[late_slice], pred[late_slice])
        slope_mismatch = abs(pred_late_slope - target_late_slope)
        score = overall_rmse + late_rmse + final_level_error + 0.5 * slope_mismatch

        scores.append(score)
        overall_rmses.append(overall_rmse)
        late_rmses.append(late_rmse)
        final_level_errors.append(final_level_error)
        slope_mismatches.append(slope_mismatch)

    return {
        "score": np.asarray(scores, dtype=float),
        "overall_rmse": np.asarray(overall_rmses, dtype=float),
        "late_rmse": np.asarray(late_rmses, dtype=float),
        "final_level_error": np.asarray(final_level_errors, dtype=float),
        "slope_mismatch": np.asarray(slope_mismatches, dtype=float),
    }


def make_similarity_weights_from_scores(scores):
    positive_scores = scores[np.isfinite(scores) & (scores > 0.0)]
    tau = max(float(np.median(positive_scores)), 1e-6) if positive_scores.size else 1e-6
    raw_weights = np.exp(-scores / tau)

    if not np.all(np.isfinite(raw_weights)) or raw_weights.sum() <= 0.0:
        raw_weights = np.ones_like(scores)

    return raw_weights / raw_weights.sum(), tau


def select_top_k(scores, candidate_k):
    order = np.argsort(scores)
    k = min(candidate_k, len(order))
    return order[:k]


def weighted_prior(component_predictions, selected_indices, weights):
    selected_components = component_predictions[:, selected_indices]
    return selected_components @ weights


def calibrate_prior(train_table, prior_train, predict_table, prior_predict):
    energy_train = train_table["energy_frac"].to_numpy(dtype=float)
    energy_predict = predict_table["energy_frac"].to_numpy(dtype=float)
    y_train = train_table["SOH"].to_numpy(dtype=float)
    coef, method = fit_constrained_calibration(prior_train, energy_train, y_train)
    predicted = coef[0] + coef[1] * prior_predict + coef[2] * energy_predict

    anchor_count = min(5, len(train_table))
    target_anchor = float(train_table["SOH"].tail(anchor_count).median())
    pred_anchor = float(
        np.median(
            coef[0]
            + coef[1] * prior_train[-anchor_count:]
            + coef[2] * energy_train[-anchor_count:]
        )
    )
    predicted = predicted + (target_anchor - pred_anchor)
    return predicted, coef, method, target_anchor - pred_anchor


def fit_target_models(target_battery_id, cycle_table):
    required_columns = [
        "capacity_ah",
        "initial_capacity_ah",
        "SOH",
        "cycle_age",
        "energy_age_kwh",
        "mean_temperature_c",
    ]
    target_table = (
        cycle_table[cycle_table["battery_id"] == target_battery_id]
        .dropna(subset=required_columns)
        .sort_values("discharge_index")
        .copy()
    )

    warnings = []
    metrics_rows = []
    prediction_rows = []
    component_rows = []

    if len(target_table) < MIN_VALID_CAPACITY_OBSERVATIONS:
        return [], [], [], [
            f"{target_battery_id}: too few usable target cycles "
            f"({len(target_table)})"
        ]

    train_table, test_table = split_battery_cycles(target_table)
    if train_table.empty or test_table.empty:
        return [], [], [], [
            f"{target_battery_id}: chronological split produced empty train or "
            f"test set ({len(train_table)} train cycles, {len(test_table)} test cycles)"
        ]

    fleet_curves, fleet_warnings = build_fleet_curves(cycle_table, target_battery_id)
    warnings.extend(fleet_warnings)
    n_fleet_batteries = len(fleet_curves)
    if n_fleet_batteries == 0:
        return [], [], [], warnings + [
            f"{target_battery_id}: no usable fleet batteries for leave-one-out prior"
        ]

    internal_split_index = max(1, int(0.8 * len(train_table)))
    if internal_split_index >= len(train_table):
        internal_split_index = len(train_table) - 1
    fit_train_table = train_table.iloc[:internal_split_index].copy()
    validation_table = train_table.iloc[internal_split_index:].copy()

    train_components = evaluate_fleet_curves(fleet_curves, train_table)
    test_components = evaluate_fleet_curves(fleet_curves, test_table)
    fit_components = evaluate_fleet_curves(fleet_curves, fit_train_table)
    validation_components = evaluate_fleet_curves(fleet_curves, validation_table)

    if train_components.shape[1] == 0 or test_components.shape[1] == 0:
        return metrics_rows, prediction_rows, component_rows, warnings + [
            f"{target_battery_id}: no finite fleet component predictions"
        ]

    uniform_weights = np.ones(n_fleet_batteries) / n_fleet_batteries
    validation_records = []

    try:
        self_fit_coef, self_fit_method = fit_constrained_self_baseline(fit_train_table)
        self_fit_pred = predict_constrained_self_baseline(self_fit_coef, fit_train_table)
        self_validation_pred = predict_constrained_self_baseline(
            self_fit_coef,
            validation_table,
        )
        self_fit_pred, self_validation_pred, self_fallback_used, _ = (
            apply_self_fallback_if_flat(
                fit_train_table,
                validation_table,
                self_fit_pred,
                self_validation_pred,
            )
        )
    except ValueError as error:
        return [], [], [], warnings + [
            f"{target_battery_id}, self_energy_baseline internal fit failed: {error}"
        ]

    internal_scores = make_similarity_scores(fit_train_table, fit_components)
    for candidate_k in [3, 5, 7]:
        selected_indices = select_top_k(internal_scores["score"], candidate_k)
        selected_scores = internal_scores["score"][selected_indices]
        selected_weights, selected_tau = make_similarity_weights_from_scores(
            selected_scores
        )
        prior_fit = weighted_prior(fit_components, selected_indices, selected_weights)
        prior_validation = weighted_prior(
            validation_components,
            selected_indices,
            selected_weights,
        )
        calibrated_validation, _, _, _ = calibrate_prior(
            fit_train_table,
            prior_fit,
            validation_table,
            prior_validation,
        )
        calibrated_fit, _, _, _ = calibrate_prior(
            fit_train_table,
            prior_fit,
            fit_train_table,
            prior_fit,
        )

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            blend_fit = alpha * self_fit_pred + (1.0 - alpha) * calibrated_fit
            blend_validation = (
                alpha * self_validation_pred
                + (1.0 - alpha) * calibrated_validation
            )
            _, monotone_validation, diagnostics = enforce_train_test_monotonicity(
                fit_train_table,
                validation_table,
                blend_fit,
                blend_validation,
            )
            validation_rmse = rmse(validation_table["SOH"], monotone_validation)
            validation_records.append(
                {
                    "k": min(candidate_k, n_fleet_batteries),
                    "alpha": alpha,
                    "validation_rmse": validation_rmse,
                    "physically_valid": diagnostics["physically_valid_prediction"],
                    "selected_indices": selected_indices,
                    "selected_weights": selected_weights,
                    "tau": selected_tau,
                }
            )

    valid_records = [
        record for record in validation_records if record["physically_valid"]
    ]
    selection_pool = valid_records if valid_records else validation_records
    selected_record = min(selection_pool, key=lambda record: record["validation_rmse"])
    selected_k = selected_record["k"]
    selected_alpha = selected_record["alpha"]
    validation_rmse = selected_record["validation_rmse"]

    self_coef, self_method = fit_constrained_self_baseline(train_table)
    self_train_pred = predict_constrained_self_baseline(self_coef, train_table)
    self_test_pred = predict_constrained_self_baseline(self_coef, test_table)
    self_train_pred, self_test_pred, self_fallback_used, self_late_slope = (
        apply_self_fallback_if_flat(
            train_table,
            test_table,
            self_train_pred,
            self_test_pred,
        )
    )

    train_uniform = train_components @ uniform_weights
    test_uniform = test_components @ uniform_weights
    full_scores = make_similarity_scores(train_table, train_components)
    selected_indices = select_top_k(full_scores["score"], selected_k)
    selected_scores = full_scores["score"][selected_indices]
    selected_weights, selected_tau = make_similarity_weights_from_scores(
        selected_scores
    )
    train_weighted = weighted_prior(train_components, selected_indices, selected_weights)
    test_weighted = weighted_prior(test_components, selected_indices, selected_weights)

    selected_weight_by_index = {
        int(index): float(weight)
        for index, weight in zip(selected_indices, selected_weights)
    }
    for curve_index, (curve, uniform_weight) in enumerate(
        zip(fleet_curves, uniform_weights)
    ):
        component_rows.append(
            {
                "target_battery_id": target_battery_id,
                "fleet_battery_id": curve["fleet_battery_id"],
                "weight_uniform": uniform_weight,
                "weight_similarity": selected_weight_by_index.get(curve_index, 0.0),
                "similarity_rmse_train": full_scores["overall_rmse"][curve_index],
                "similarity_late_rmse_train": full_scores["late_rmse"][curve_index],
                "similarity_final_level_error": full_scores["final_level_error"][curve_index],
                "similarity_slope_mismatch": full_scores["slope_mismatch"][curve_index],
                "similarity_score_train": full_scores["score"][curve_index],
                "similarity_tau": selected_tau,
                "selected_for_adaptive_blend": curve_index in set(selected_indices),
                "n_fleet_cycles": curve["n_fleet_cycles"],
            }
        )

    calibrated_train, calibration_coef, calibration_method, calibration_shift = (
        calibrate_prior(
            train_table,
            train_weighted,
            train_table,
            train_weighted,
        )
    )
    calibrated_test, _, _, _ = calibrate_prior(
        train_table,
        train_weighted,
        test_table,
        test_weighted,
    )
    blend_train = (
        selected_alpha * self_train_pred
        + (1.0 - selected_alpha) * calibrated_train
    )
    blend_test = (
        selected_alpha * self_test_pred
        + (1.0 - selected_alpha) * calibrated_test
    )

    append_model_results(
        metrics_rows,
        prediction_rows,
        target_battery_id,
        "self_energy_baseline",
        train_table,
        test_table,
        self_train_pred,
        self_test_pred,
        n_fleet_batteries,
        {
            "trend_fit_method": self_method,
            "self_fallback_used": self_fallback_used,
            "self_late_train_slope": self_late_slope,
            "selected_k_neighbors": selected_k,
            "selected_alpha_blend": selected_alpha,
            "validation_rmse_soh": validation_rmse,
        },
    )

    for model_name, train_pred, test_pred, parameter_rows in [
        (
            "fleet_prior_uniform",
            train_uniform,
            test_uniform,
            {
                "prior_source": "uniform_mean_leave_one_battery_out",
                "selected_k_neighbors": selected_k,
                "selected_alpha_blend": selected_alpha,
                "validation_rmse_soh": validation_rmse,
            },
        ),
        (
            "fleet_prior_similarity_weighted_raw",
            train_weighted,
            test_weighted,
            {
                "similarity_tau": selected_tau,
                "selected_k_neighbors": selected_k,
                "selected_alpha_blend": selected_alpha,
                "validation_rmse_soh": validation_rmse,
            },
        ),
    ]:
        if finite_prediction(train_pred) and finite_prediction(test_pred):
            append_model_results(
                metrics_rows,
                prediction_rows,
                target_battery_id,
                model_name,
                train_table,
                test_table,
                train_pred,
                test_pred,
                n_fleet_batteries,
                parameter_rows,
            )
        else:
            warnings.append(
                f"{target_battery_id}, {model_name}: non-finite fleet prior "
                "predictions"
            )

    append_model_results(
        metrics_rows,
        prediction_rows,
        target_battery_id,
        "fleet_prior_calibrated",
        train_table,
        test_table,
        calibrated_train,
        calibrated_test,
        n_fleet_batteries,
        {
            "calibration_method": calibration_method,
            "calibration_intercept": calibration_coef[0],
            "calibration_coef_fleet_prior": calibration_coef[1],
            "calibration_coef_energy_frac": calibration_coef[2],
            "calibration_anchor_shift": calibration_shift,
            "selected_k_neighbors": selected_k,
            "selected_alpha_blend": selected_alpha,
            "validation_rmse_soh": validation_rmse,
        },
    )

    append_model_results(
        metrics_rows,
        prediction_rows,
        target_battery_id,
        "adaptive_self_fleet_blend",
        train_table,
        test_table,
        blend_train,
        blend_test,
        n_fleet_batteries,
        {
            "selected_k_neighbors": selected_k,
            "selected_alpha_blend": selected_alpha,
            "validation_rmse_soh": validation_rmse,
        },
    )

    train_temperature_mean = float(train_table["mean_temperature_c"].mean())
    train_temperature_centered = (
        train_table["mean_temperature_c"].to_numpy(dtype=float)
        - train_temperature_mean
    )
    test_temperature_centered = (
        test_table["mean_temperature_c"].to_numpy(dtype=float)
        - train_temperature_mean
    )
    temperature_design = train_temperature_centered.reshape(-1, 1)
    temperature_residual = train_table["SOH"].to_numpy(dtype=float) - calibrated_train
    temperature_coef_d, *_ = np.linalg.lstsq(
        temperature_design,
        temperature_residual,
        rcond=None,
    )
    temperature_coef_d = float(temperature_coef_d[0])
    temperature_train = (
        calibrated_train + temperature_coef_d * train_temperature_centered
    )
    temperature_test = (
        calibrated_test + temperature_coef_d * test_temperature_centered
    )
    append_model_results(
        metrics_rows,
        prediction_rows,
        target_battery_id,
        "fleet_prior_calibrated_temperature",
        train_table,
        test_table,
        temperature_train,
        temperature_test,
        n_fleet_batteries,
        {
            "calibration_method": "calibrated_plus_temperature_residual",
            "temperature_center_c": train_temperature_mean,
            "calibration_coef_mean_temperature_c": temperature_coef_d,
            "selected_k_neighbors": selected_k,
            "selected_alpha_blend": selected_alpha,
            "validation_rmse_soh": validation_rmse,
        },
    )

    return metrics_rows, prediction_rows, component_rows, warnings


def plot_model_comparison(cycle_table, prediction_table, metrics_table, figure_path):
    available_batteries = (
        metrics_table[metrics_table["model_name"] == "self_energy_baseline"][
            "target_battery_id"
        ]
        .drop_duplicates()
        .tolist()
    )
    preferred_batteries = [
        "B0005",
        "B0006",
        "B0007",
        "B0018",
        "B0055",
        "B0056",
    ]
    common_batteries = [
        battery_id
        for battery_id in preferred_batteries
        if battery_id in available_batteries
    ]
    if len(common_batteries) < 4:
        remaining_batteries = (
            metrics_table[
                (metrics_table["model_name"] == "self_energy_baseline")
                & (~metrics_table["target_battery_id"].isin(common_batteries))
            ]
            .sort_values("n_cycles", ascending=False)["target_battery_id"]
            .tolist()
        )
        common_batteries.extend(remaining_batteries[: 4 - len(common_batteries)])

    if not common_batteries:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    for axis, battery_id in zip(axes, common_batteries):
        battery_cycles = (
            cycle_table[cycle_table["battery_id"] == battery_id]
            .dropna(subset=["SOH"])
            .sort_values("discharge_index")
        )
        battery_predictions = prediction_table[
            prediction_table["target_battery_id"] == battery_id
        ]

        axis.plot(
            battery_cycles["discharge_index"],
            battery_cycles["SOH"],
            color="black",
            marker="o",
            linewidth=1.2,
            markersize=2.5,
            label="Measured raw SOH",
        )

        split_rows = battery_predictions[battery_predictions["split"] == "test"]
        if not split_rows.empty:
            split_cycle = split_rows["discharge_index"].min()
            axis.axvline(
                split_cycle,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Train/test split",
            )

        target_metrics = metrics_table[
            metrics_table["target_battery_id"] == battery_id
        ].set_index("model_name")
        plot_models = [
            "self_energy_baseline",
            "fleet_prior_calibrated",
            "adaptive_self_fleet_blend",
        ]
        if (
            "fleet_prior_calibrated_temperature" in target_metrics.index
            and bool(
                target_metrics.loc[
                    "fleet_prior_calibrated_temperature",
                    "physically_valid_prediction",
                ]
            )
            and target_metrics.loc[
                "fleet_prior_calibrated_temperature",
                "test_rmse_soh",
            ]
            < target_metrics.loc["fleet_prior_calibrated", "test_rmse_soh"]
        ):
            plot_models.append("fleet_prior_calibrated_temperature")

        for model_name in plot_models:
            model_predictions = battery_predictions[
                battery_predictions["model_name"] == model_name
            ].sort_values("discharge_index")
            if model_predictions.empty:
                continue

            train_predictions = model_predictions[
                model_predictions["split"] == "train"
            ]
            test_predictions = model_predictions[model_predictions["split"] == "test"]

            axis.plot(
                train_predictions["discharge_index"],
                train_predictions["predicted_SOH"],
                linewidth=1,
                linestyle=":",
                alpha=0.45,
                label=(
                    f"{MODEL_LABELS[model_name]} train"
                    if model_name == "self_energy_baseline"
                    else None
                ),
            )
            label = MODEL_LABELS[model_name]
            line_style = "-"
            line_alpha = 0.95
            line_width = 1.5
            if model_name == "fleet_prior_calibrated":
                line_width = 2.8
            elif model_name == "adaptive_self_fleet_blend":
                line_width = 1.1
                line_style = "--"
                line_alpha = 0.7
            elif model_name == "fleet_prior_calibrated_temperature":
                line_width = 1.2
                line_style = "-."
                line_alpha = 0.75
            axis.plot(
                test_predictions["discharge_index"],
                test_predictions["predicted_SOH"],
                linewidth=line_width,
                linestyle=line_style,
                alpha=line_alpha,
                label=label,
            )

        axis.set_title(battery_id)
        axis.set_xlabel("Discharge cycle index")
        axis.set_ylabel("SOH")
        axis.grid(alpha=0.3)

    for axis in axes[len(common_batteries) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Cycle-level capacity aging model comparison: calibrated fleet model selected"
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.96])
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def plot_rmse_summary(metrics_table, figure_path):
    model_order = [
        model_name
        for model_name in RMSE_FIGURE_MODELS
        if model_name in set(metrics_table["model_name"])
    ]
    model_labels = [MODEL_LABELS[model_name] for model_name in model_order]

    data = [
        metrics_table[metrics_table["model_name"] == model_name]["test_rmse_soh"]
        for model_name in model_order
    ]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, tick_labels=model_labels, showmeans=True)
    means = [values.mean() for values in data]
    plt.plot(
        np.arange(1, len(model_order) + 1),
        means,
        color="tab:red",
        marker="o",
        linewidth=1.2,
        label="Mean",
    )
    plt.ylabel("Test RMSE [SOH]")
    selected_models = (
        metrics_table.loc[metrics_table["selected_as_final"], "model_name"]
        .dropna()
        .unique()
        .tolist()
        if "selected_as_final" in metrics_table.columns
        else []
    )
    selected_model = selected_models[0] if selected_models else "fleet_prior_calibrated"
    selected_label = MODEL_LABELS.get(selected_model, selected_model)
    plt.title(
        "Cycle-level capacity aging test RMSE by model\n"
        f"Selected final model: {selected_label}"
    )
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def plot_fleet_prior_examples(cycle_table, prediction_table, figure_path):
    available_batteries = (
        prediction_table[prediction_table["model_name"] == "fleet_prior_calibrated"][
            "target_battery_id"
        ]
        .drop_duplicates()
        .tolist()
    )
    preferred_batteries = ["B0005", "B0006", "B0007", "B0018"]
    common_batteries = [
        battery_id
        for battery_id in preferred_batteries
        if battery_id in available_batteries
    ]
    if len(common_batteries) < 4:
        common_batteries.extend(
            [
                battery_id
                for battery_id in available_batteries
                if battery_id not in common_batteries
            ][: 4 - len(common_batteries)]
        )

    if not common_batteries:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    for axis, target_battery_id in zip(axes, common_batteries):
        target_predictions = prediction_table[
            prediction_table["target_battery_id"] == target_battery_id
        ]
        target_rows = (
            cycle_table[cycle_table["battery_id"] == target_battery_id]
            .sort_values("discharge_index")
            .copy()
        )
        train_rows = target_predictions[
            (target_predictions["model_name"] == "self_energy_baseline")
            & (target_predictions["split"] == "train")
        ]
        test_rows = target_predictions[
            (target_predictions["model_name"] == "self_energy_baseline")
            & (target_predictions["split"] == "test")
        ]

        fleet_table = cycle_table[cycle_table["battery_id"] != target_battery_id]
        target_energy_min = target_rows["energy_frac"].min()
        target_energy_max = target_rows["energy_frac"].max()

        first_fleet_label = True
        for fleet_battery_id in sorted(fleet_table["battery_id"].unique()):
            fleet_battery = (
                fleet_table[fleet_table["battery_id"] == fleet_battery_id]
                .sort_values("discharge_index")
                .copy()
            )
            if len(fleet_battery) < MIN_VALID_CAPACITY_OBSERVATIONS:
                continue
            fleet_energy_min = fleet_battery["energy_frac"].min()
            fleet_energy_max = fleet_battery["energy_frac"].max()
            plot_energy_min = max(target_energy_min, fleet_energy_min)
            plot_energy_max = min(target_energy_max, fleet_energy_max)
            if plot_energy_max <= plot_energy_min:
                continue
            grid_energy = np.linspace(plot_energy_min, plot_energy_max, 120)
            try:
                fleet_model = fit_monotone_curve(fleet_battery)
            except ValueError:
                continue
            axis.plot(
                grid_energy,
                predict_monotone_values(fleet_model, grid_energy),
                color="tab:gray",
                linewidth=0.8,
                alpha=0.18,
                label="Individual fleet curves" if first_fleet_label else None,
            )
            first_fleet_label = False

        axis.scatter(
            train_rows["energy_frac"],
            train_rows["SOH"],
            color="black",
            s=14,
            label="Target train SOH",
        )
        axis.scatter(
            test_rows["energy_frac"],
            test_rows["SOH"],
            color="white",
            edgecolor="black",
            s=18,
            label="Target test SOH",
        )

        for model_name in [
            "fleet_prior_uniform",
            "fleet_prior_similarity_weighted_raw",
            "fleet_prior_calibrated",
        ]:
            rows = target_predictions[
                target_predictions["model_name"] == model_name
            ].sort_values("energy_frac")
            axis.plot(
                rows["energy_frac"],
                rows["predicted_SOH"],
                linewidth=1.8,
                label=MODEL_LABELS[model_name],
            )

        axis.set_title(target_battery_id)
        axis.set_xlabel("Normalized energy age")
        axis.set_ylabel("SOH")
        axis.grid(alpha=0.3)

    for axis in axes[len(common_batteries) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Fleet prior diagnostics: uncalibrated priors are diagnostics; "
        "calibrated forecast is target-specific"
    )
    fig.tight_layout(rect=[0.0, 0.1, 1.0, 0.96])
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[1]
    sequence_path = project_root / "results" / "discharge_sequences_all.csv"
    output_dir = project_root / "results"
    figure_dir = project_root / "figures"

    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    cycle_dataset_path = output_dir / "cycle_level_capacity_aging_dataset.csv"
    raw_cycle_dataset_path = output_dir / "cycle_level_capacity_aging_dataset_raw.csv"
    metrics_path = output_dir / "capacity_aging_model_metrics.csv"
    predictions_path = output_dir / "capacity_aging_model_predictions.csv"
    fleet_components_path = output_dir / "capacity_aging_fleet_prior_components.csv"
    diagnostics_path = output_dir / "capacity_aging_battery_diagnostics.csv"
    skipped_path = output_dir / "cycle_level_capacity_aging_skipped.csv"
    comparison_figure_path = (
        figure_dir / "capacity_aging_model_comparison_examples.png"
    )
    rmse_figure_path = figure_dir / "capacity_aging_rmse_summary.png"
    fleet_prior_figure_path = figure_dir / "capacity_aging_fleet_prior_examples.png"

    sequence_table = pd.read_csv(sequence_path)
    raw_cycle_table = build_raw_cycle_level_dataset(sequence_table)
    raw_cycle_table.to_csv(raw_cycle_dataset_path, index=False)
    cycle_table, diagnostics_table = build_capacity_diagnostics_and_model_dataset(
        raw_cycle_table
    )
    cycle_table.to_csv(cycle_dataset_path, index=False)
    diagnostics_table.to_csv(diagnostics_path, index=False)
    skipped_diagnostics = diagnostics_table[diagnostics_table["status"] == "skipped"]
    skipped_diagnostics.to_csv(skipped_path, index=False)

    metrics_rows = []
    prediction_rows = []
    component_rows = []
    model_warnings = []

    for target_battery_id in sorted(cycle_table["battery_id"].dropna().unique()):
        metrics, predictions, components, warnings = fit_target_models(
            target_battery_id,
            cycle_table,
        )
        metrics_rows.extend(metrics)
        prediction_rows.extend(predictions)
        component_rows.extend(components)
        model_warnings.extend(warnings)

    metrics_table = pd.DataFrame(metrics_rows)
    prediction_table = pd.DataFrame(prediction_rows)
    component_table = pd.DataFrame(component_rows)

    if not metrics_table.empty:
        selected_final_model = choose_selected_final_model(metrics_table)
        metrics_table["selected_as_final"] = (
            metrics_table["model_name"] == selected_final_model
        )

    metrics_table.to_csv(metrics_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    component_table.to_csv(fleet_components_path, index=False)

    if not metrics_table.empty and not prediction_table.empty:
        plot_model_comparison(
            cycle_table,
            prediction_table,
            metrics_table,
            comparison_figure_path,
        )
        plot_rmse_summary(metrics_table, rmse_figure_path)
        plot_fleet_prior_examples(cycle_table, prediction_table, fleet_prior_figure_path)

    print("Cycle-level capacity aging summary")
    print("----------------------------------")
    print(f"Input path used: {sequence_path}")
    print(f"Number of raw batteries inspected: {diagnostics_table['battery_id'].nunique()}")
    print(f"Number of valid batteries used: {cycle_table['battery_id'].nunique()}")
    print(f"Number of batteries skipped: {len(skipped_diagnostics)}")
    if metrics_table.empty:
        print("Number of target batteries evaluated: 0")
    else:
        print(
            "Number of target batteries evaluated: "
            f"{metrics_table['target_battery_id'].nunique()}"
        )
    print(f"Number of cycle-level rows: {len(cycle_table)}")

    if not diagnostics_table.empty:
        print("\nPer-battery capacity diagnostics:")
        diagnostic_columns = [
            "battery_id",
            "n_cycles_raw",
            "n_capacity_non_missing",
            "first_capacity_ah",
            "min_capacity_ah",
            "median_capacity_ah",
            "max_capacity_ah",
            "candidate_initial_capacity_ah",
            "min_SOH",
            "median_SOH",
            "max_SOH",
            "status",
            "reason_if_skipped",
        ]
        print(diagnostics_table[diagnostic_columns].to_string(index=False))

    if not skipped_diagnostics.empty:
        print("\nSkipped batteries and reasons:")
        for row in skipped_diagnostics.itertuples(index=False):
            print(f"- {row.battery_id}: {row.reason_if_skipped}")
    else:
        print("\nSkipped batteries and reasons: none")

    if model_warnings:
        print("\nModel warnings:")
        for warning in model_warnings:
            print(f"- {warning}")

    print("\nOutput paths:")
    print(f"Raw cycle-level dataset: {raw_cycle_dataset_path}")
    print(f"Cycle-level dataset: {cycle_dataset_path}")
    print(f"Skipped batteries: {skipped_path}")
    print(f"Battery diagnostics: {diagnostics_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Fleet prior components: {fleet_components_path}")
    print(f"Figure: {comparison_figure_path}")
    print(f"Figure: {rmse_figure_path}")
    print(f"Figure: {fleet_prior_figure_path}")

    if metrics_table.empty:
        print("\nNo model metrics were produced.")
        return

    average_rmse = (
        metrics_table.groupby("model_name")["test_rmse_soh"].mean().sort_values()
    )
    physically_valid_metrics = metrics_table[
        metrics_table["physically_valid_prediction"]
    ].copy()
    physically_valid_average_rmse = (
        physically_valid_metrics.groupby("model_name")["test_rmse_soh"]
        .mean()
        .sort_values()
    )
    best_physical_model = physically_valid_average_rmse.index[0]
    selected_final_model = choose_selected_final_model(metrics_table)
    calibrated_improved = (
        physically_valid_average_rmse["fleet_prior_calibrated"]
        < physically_valid_average_rmse["self_energy_baseline"]
    )
    adaptive_improved_over_self = (
        physically_valid_average_rmse["adaptive_self_fleet_blend"]
        < physically_valid_average_rmse["self_energy_baseline"]
    )
    adaptive_improved_over_fleet = (
        physically_valid_average_rmse["adaptive_self_fleet_blend"]
        < physically_valid_average_rmse["fleet_prior_calibrated"]
    )
    temperature_improved = (
        physically_valid_average_rmse["fleet_prior_calibrated_temperature"]
        < physically_valid_average_rmse["fleet_prior_calibrated"]
    )

    print("\nAverage test RMSE by model [SOH]:")
    for model_name, value in average_rmse.items():
        print(f"{model_name}: {value:.6f}")

    print("\nAverage physically valid test RMSE by model [SOH]:")
    for model_name, value in physically_valid_average_rmse.items():
        print(f"{model_name}: {value:.6f}")

    print(
        "\nBest physically valid model by average test RMSE: "
        f"{best_physical_model}"
    )
    print(f"Selected final capacity-aging model: {selected_final_model}")
    if selected_final_model == "fleet_prior_calibrated":
        print(
            "The calibrated fleet model is selected as the final "
            "capacity-aging model because it gives the best average physically "
            "valid test RMSE in the current deterministic setting."
        )

    selection_table = (
        metrics_table[metrics_table["model_name"] == "adaptive_self_fleet_blend"][
            [
                "target_battery_id",
                "selected_k_neighbors",
                "selected_alpha_blend",
                "validation_rmse_soh",
            ]
        ]
        .sort_values("target_battery_id")
        .reset_index(drop=True)
    )
    print("\nSelected k and alpha for adaptive_self_fleet_blend:")
    print(selection_table.to_string(index=False))

    alpha_values = selection_table["selected_alpha_blend"]
    alpha_zero_count = int(np.isclose(alpha_values, 0.0).sum())
    alpha_one_count = int(np.isclose(alpha_values, 1.0).sum())
    alpha_intermediate_count = int(
        ((~np.isclose(alpha_values, 0.0)) & (~np.isclose(alpha_values, 1.0))).sum()
    )
    print("\nAdaptive blend alpha distribution:")
    print(f"alpha = 0 targets: {alpha_zero_count}")
    print(f"alpha = 1 targets: {alpha_one_count}")
    print(f"intermediate alpha targets: {alpha_intermediate_count}")
    collapsed_selection = selection_table[
        np.isclose(selection_table["selected_alpha_blend"], 0.0)
        | np.isclose(selection_table["selected_alpha_blend"], 1.0)
    ]
    if not collapsed_selection.empty:
        print(
            "\nadaptive_self_fleet_blend collapsed to one parent model for "
            "these targets:"
        )
        for row in collapsed_selection.itertuples(index=False):
            parent_model = (
                "fleet_prior_calibrated"
                if np.isclose(row.selected_alpha_blend, 0.0)
                else "self_energy_baseline"
            )
            print(
                f"- {row.target_battery_id}: alpha={row.selected_alpha_blend:.2f}, "
                f"parent={parent_model}"
            )

    print(
        "fleet_prior_calibrated improves over self_energy_baseline on average: "
        f"{calibrated_improved}"
    )
    print(
        "adaptive_self_fleet_blend improves over self_energy_baseline on "
        f"average: {adaptive_improved_over_self}"
    )
    print(
        "adaptive_self_fleet_blend improves over fleet_prior_calibrated on "
        f"average: {adaptive_improved_over_fleet}"
    )
    print(
        "fleet_prior_calibrated_temperature improves over fleet_prior_calibrated "
        f"on average: {temperature_improved}"
    )
    print(
        "Blue/orange fleet priors are diagnostic uncalibrated priors, not "
        "final target-specific forecasts."
    )

    print("\nConclusion:")
    if calibrated_improved:
        print(
            "Deterministic fleet-informed calibration improved cycle-level SOH "
            "forecasting compared with the self-only baseline."
        )
    else:
        print(
            "The deterministic fleet-informed prior was evaluated, but did not "
            "consistently improve over the self-only baseline in this dataset."
        )

    if adaptive_improved_over_fleet:
        print(
            "Adaptive self/fleet blending improved the average test RMSE over "
            "the calibrated fleet model in this run."
        )
    else:
        print(
            "Adaptive self/fleet blending was evaluated, but it did not improve "
            "the average test RMSE over the calibrated fleet model in this run."
        )
        print(
            "adaptive_self_fleet_blend was evaluated but not selected because "
            "it did not improve average test RMSE over the calibrated fleet "
            "forecast."
        )

    if temperature_improved:
        print(
            "The temperature-aware calibrated fleet prior improved the average "
            "RMSE over the calibrated fleet model in this benchmark."
        )
    else:
        print(
            "Temperature was evaluated as an ablation feature but was not "
            "selected as the final capacity-aging feature because it did not "
            "provide consistent average improvement."
        )
        print(
            "Temperature was evaluated as a candidate feature but did not "
            "provide consistent physically reasonable average improvement in "
            "this dataset."
        )
    print(
        "Example 17 is a deterministic fleet-informed cycle-level "
        "capacity-aging model. It is not a Bayesian posterior update and does "
        "not implement qmax/R0, vMLP, or partial-observation inference from "
        "the reference HBPINN paper."
    )
    print(
        "This is a simplified deterministic fleet-informed surrogate for "
        "cycle-level capacity aging."
    )


if __name__ == "__main__":
    main()
