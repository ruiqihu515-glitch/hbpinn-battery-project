from pathlib import Path
import os

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

CYCLE_DATASET_PATH = RESULTS_DIR / "capacity_aging_v2_cycle_dataset.csv"
PREDICTIONS_PATH = RESULTS_DIR / "capacity_aging_v2_predictions.csv"
METRICS_PATH = RESULTS_DIR / "capacity_aging_v2_metrics.csv"
FLEET_COMPONENTS_PATH = RESULTS_DIR / "capacity_aging_v2_fleet_components.csv"

FLEET_UPDATE_FIGURE_PATH = (
    FIGURES_DIR / "capacity_aging_v2_fleet_update_examples.png"
)
MODEL_COMPARISON_FIGURE_PATH = (
    FIGURES_DIR / "capacity_aging_v2_model_comparison_examples.png"
)
RMSE_SUMMARY_FIGURE_PATH = FIGURES_DIR / "capacity_aging_v2_rmse_summary.png"

REQUIRED_COLUMNS = [
    "battery_id",
    "discharge_index",
    "temperature_c",
    "capacity_ah",
    "cycle_energy_kwh",
]

TARGET_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
MIN_VALID_CYCLES = 20
MAX_ALLOWED_SOH = 1.5
MAX_ALLOWED_MEDIAN_SOH = 1.2
MIN_ALLOWED_SOH = 0.3
PREDICTION_INCREASE_TOLERANCE = 0.003
MODEL_NAMES = [
    "self_only_energy_baseline",
    "fleet_prior_weighted_raw",
    "fleet_updated_deterministic",
]


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


def safe_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def sorted_unique_xy(x, y):
    table = (
        pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("x")
    )
    if table.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    grouped = table.groupby("x", as_index=False)["y"].mean()
    return grouped["x"].to_numpy(dtype=float), grouped["y"].to_numpy(dtype=float)


class MonotoneAgingCurve:
    def __init__(self, x_train, y_train):
        x, y = sorted_unique_xy(x_train, y_train)
        if len(x) < 2:
            raise ValueError("At least two unique energy points are required.")

        isotonic = IsotonicRegression(increasing=False, out_of_bounds="clip")
        y_iso = isotonic.fit_transform(x, y)
        y_iso = np.minimum.accumulate(y_iso)

        self.x = x
        self.y = y_iso
        self.x_min = float(x[0])
        self.x_max = float(x[-1])
        self.y_min_x = float(y_iso[0])
        self.y_max_x = float(y_iso[-1])
        self.late_slope = self._estimate_late_slope()

        if PchipInterpolator is not None and len(x) >= 3:
            self.interpolator = PchipInterpolator(x, y_iso, extrapolate=False)
        else:
            self.interpolator = None

    def _estimate_late_slope(self):
        count = min(max(3, int(np.ceil(0.3 * len(self.x)))), len(self.x))
        x_late = self.x[-count:]
        y_late = self.y[-count:]
        if len(np.unique(x_late)) < 2:
            return 0.0
        return min(float(np.polyfit(x_late, y_late, 1)[0]), 0.0)

    def __call__(self, x_eval):
        x_eval = np.asarray(x_eval, dtype=float)
        flat = x_eval.reshape(-1)
        clipped = np.clip(flat, self.x_min, self.x_max)

        if self.interpolator is not None:
            pred = np.asarray(self.interpolator(clipped), dtype=float)
        else:
            pred = np.interp(clipped, self.x, self.y)

        before = flat < self.x_min
        after = flat > self.x_max
        if np.any(before):
            pred[before] = self.y_min_x
        if np.any(after):
            dx = flat[after] - self.x_max
            span = max(self.x_max - self.x_min, 1e-6)
            mild_curve = 1.0 + 0.25 * np.minimum(dx / span, 1.0)
            pred[after] = self.y_max_x + self.late_slope * dx * mild_curve

        order = np.argsort(flat)
        pred_sorted = pred[order]
        pred_sorted = np.minimum.accumulate(pred_sorted)
        restored = np.empty_like(pred_sorted)
        restored[order] = pred_sorted
        return restored.reshape(x_eval.shape)


def build_cycle_dataset(sequence_table):
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in sequence_table.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

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
            initial_capacity_ah = float(positive_capacity.iloc[:3].median())
        elif len(positive_capacity) > 0:
            initial_capacity_ah = float(positive_capacity.iloc[0])
        else:
            initial_capacity_ah = np.nan

        battery_table["initial_capacity_ah"] = initial_capacity_ah
        battery_table["SOH"] = battery_table["capacity_ah"] / initial_capacity_ah

        finite_soh = battery_table["SOH"][
            np.isfinite(battery_table["SOH"])
        ].to_numpy(dtype=float)
        reasons = []
        essential = [
            "capacity_ah",
            "cycle_energy_kwh",
            "mean_temperature_c",
            "min_temperature_c",
            "max_temperature_c",
            "number_of_steps",
        ]
        if len(positive_capacity) < MIN_VALID_CYCLES:
            reasons.append("too few valid cycles")
        if not np.isfinite(initial_capacity_ah) or initial_capacity_ah <= 0.0:
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

        battery_table = battery_table.dropna(subset=essential + ["SOH"]).copy()
        battery_table = battery_table[
            np.isfinite(battery_table["SOH"])
            & np.isfinite(battery_table["cycle_energy_kwh"])
            & (battery_table["cycle_energy_kwh"] >= 0.0)
        ].copy()
        if len(battery_table) < MIN_VALID_CYCLES:
            skipped[battery_id] = "too few valid cycles after numeric filtering"
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
        "mean_temperature_c",
        "min_temperature_c",
        "max_temperature_c",
        "number_of_steps",
    ]
    return pd.concat(valid_tables, ignore_index=True)[output_columns], skipped


def split_target(table):
    table = table.sort_values("discharge_index").reset_index(drop=True)
    split_index = int(np.floor(0.7 * len(table)))
    split_index = min(max(split_index, 2), len(table) - 1)
    train = table.iloc[:split_index].copy()
    test = table.iloc[split_index:].copy()
    return train, test


def late_count(table):
    return min(max(3, int(np.ceil(0.3 * len(table)))), len(table))


def estimate_late_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(x)) < 2:
        return 0.0
    return min(float(np.polyfit(x, y, 1)[0]), 0.0)


def score_fleet_curve(curve, train_table):
    x_train = train_table["energy_frac"].to_numpy(dtype=float)
    y_train = train_table["SOH"].to_numpy(dtype=float)
    pred_train = curve(x_train)

    n_late = late_count(train_table)
    overall_train_rmse = rmse(y_train, pred_train)
    late_train_rmse = rmse(y_train[-n_late:], pred_train[-n_late:])
    final_level_error = abs(
        float(np.median(y_train[-min(5, len(y_train)) :]))
        - float(np.median(pred_train[-min(5, len(pred_train)) :]))
    )
    target_slope = estimate_late_slope(x_train[-n_late:], y_train[-n_late:])
    fleet_slope = estimate_late_slope(x_train[-n_late:], pred_train[-n_late:])
    slope_mismatch = abs(target_slope - fleet_slope)
    similarity_score = (
        overall_train_rmse + late_train_rmse + final_level_error + 0.5 * slope_mismatch
    )
    return {
        "similarity_score": float(similarity_score),
        "overall_train_rmse": float(overall_train_rmse),
        "late_train_rmse": float(late_train_rmse),
        "final_level_error": float(final_level_error),
        "slope_mismatch": float(slope_mismatch),
    }


def weighted_prior(curves, component_table, x_eval):
    selected = component_table[component_table["selected_top_k"]].copy()
    if selected.empty:
        selected = component_table.nsmallest(1, "similarity_score").copy()
    pred = np.zeros(len(x_eval), dtype=float)
    for row in selected.itertuples(index=False):
        pred += float(row.weight) * curves[row.fleet_battery_id](x_eval)
    return np.minimum.accumulate(pred)


def build_components(target_id, train_table, fleet_curves, k):
    rows = []
    for fleet_battery_id, curve in fleet_curves.items():
        score = score_fleet_curve(curve, train_table)
        rows.append(
            {
                "target_battery_id": target_id,
                "fleet_battery_id": fleet_battery_id,
                **score,
            }
        )

    components = pd.DataFrame(rows).sort_values("similarity_score").reset_index(
        drop=True
    )
    k = min(k, len(components))
    selected_index = components.index[:k]
    selected_scores = components.loc[selected_index, "similarity_score"].to_numpy(
        dtype=float
    )
    positive_scores = selected_scores[selected_scores > 0.0]
    tau = max(float(np.median(positive_scores)) if positive_scores.size else 1.0, 1e-6)
    raw_weights = np.exp(-selected_scores / tau)
    weights = raw_weights / raw_weights.sum()

    components["selected_top_k"] = False
    components["weight"] = 0.0
    components.loc[selected_index, "selected_top_k"] = True
    components.loc[selected_index, "weight"] = weights
    return components


def choose_k_by_train_validation(target_id, train_table, fleet_curves):
    if len(fleet_curves) <= 3 or len(train_table) < 10:
        return min(5, max(1, len(fleet_curves)))

    inner_split = int(np.floor(0.7 * len(train_table)))
    fit_train = train_table.iloc[:inner_split].copy()
    validation = train_table.iloc[inner_split:].copy()
    x_val = validation["energy_frac"].to_numpy(dtype=float)
    y_val = validation["SOH"].to_numpy(dtype=float)

    best_k = None
    best_rmse = np.inf
    for k in [3, 5, 7]:
        if k > len(fleet_curves):
            continue
        components = build_components(target_id, fit_train, fleet_curves, k)
        pred = weighted_prior(fleet_curves, components, x_val)
        value = rmse(y_val, pred)
        if value < best_rmse:
            best_rmse = value
            best_k = k
    return best_k if best_k is not None else min(5, len(fleet_curves))


def residual_update_curve(train_x, residual_train, eval_x):
    train_x = np.asarray(train_x, dtype=float)
    residual_train = np.asarray(residual_train, dtype=float)
    eval_x = np.asarray(eval_x, dtype=float)
    order = np.argsort(train_x)
    train_x = train_x[order]
    residual_train = residual_train[order]
    x_unique, residual_unique = sorted_unique_xy(train_x, residual_train)

    if len(x_unique) >= 4 and PchipInterpolator is not None:
        interpolator = PchipInterpolator(x_unique, residual_unique, extrapolate=False)
        residual_eval = np.asarray(interpolator(np.clip(eval_x, x_unique[0], x_unique[-1])))
    else:
        residual_eval = np.interp(eval_x, x_unique, residual_unique)

    late_n = min(5, len(x_unique))
    late_slope = estimate_late_slope(x_unique[-late_n:], residual_unique[-late_n:])
    anchor_x = float(x_unique[-1])
    anchor_residual = float(residual_unique[-1])
    future = eval_x > anchor_x
    if np.any(future):
        dx = eval_x[future] - anchor_x
        span = max(anchor_x - float(x_unique[0]), 1e-6)
        damping = 1.0 / (1.0 + 1.5 * dx / span)
        residual_eval[future] = anchor_residual + late_slope * dx * damping
    return residual_eval


def enforce_final_prediction_rules(train_table, test_table, pred_all):
    pred_all = np.asarray(pred_all, dtype=float).copy()
    n_train = len(train_table)
    anchor_n = min(5, len(train_table))
    target_anchor = float(train_table["SOH"].tail(anchor_n).median())
    pred_anchor = float(np.median(pred_all[n_train - anchor_n : n_train]))
    shift = target_anchor - pred_anchor
    pred_all += shift

    if len(test_table) > 0 and pred_all[n_train] > target_anchor + 0.003:
        pred_all[n_train:] -= pred_all[n_train] - (target_anchor + 0.003)

    pred_all = np.minimum.accumulate(pred_all)
    pred_all[n_train:] = np.minimum.accumulate(pred_all[n_train:])
    if n_train > 0 and len(test_table) > 0:
        pred_all[n_train:] = np.minimum(pred_all[n_train:], pred_all[n_train - 1] + 0.003)
    return pred_all


def physical_diagnostics(predicted):
    values = np.asarray(predicted, dtype=float)
    if len(values) < 2:
        return 0, 0.0, True
    increases = np.diff(values)
    positive = increases[increases > PREDICTION_INCREASE_TOLERANCE]
    max_increase = float(positive.max()) if positive.size else 0.0
    return int(positive.size), max_increase, positive.size == 0


def prediction_rows(target_id, full_table, split_labels, model_name, model_role, pred):
    table = full_table.copy()
    table["target_battery_id"] = target_id
    table["split"] = split_labels
    table["model_name"] = model_name
    table["model_role"] = model_role
    table["predicted_SOH"] = pred
    table["predicted_capacity_ah"] = (
        table["predicted_SOH"] * table["initial_capacity_ah"]
    )
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
            "predicted_SOH",
            "predicted_capacity_ah",
        ]
    ]


def metric_row(
    target_id,
    model_name,
    model_role,
    full_table,
    split_labels,
    pred,
    n_fleet_batteries,
    top_k,
    top_ids,
):
    split_labels = np.asarray(split_labels)
    y = full_table["SOH"].to_numpy(dtype=float)
    train_mask = split_labels == "train"
    test_mask = split_labels == "test"
    train_rmse = rmse(y[train_mask], pred[train_mask])
    test_rmse = rmse(y[test_mask], pred[test_mask])
    train_mae = safe_mae(y[train_mask], pred[train_mask])
    test_mae = safe_mae(y[test_mask], pred[test_mask])
    first_test_jump = (
        float(pred[np.where(test_mask)[0][0]] - pred[np.where(train_mask)[0][-1]])
        if test_mask.any() and train_mask.any()
        else np.nan
    )
    n_increases, max_increase, physically_valid = physical_diagnostics(pred)
    selected_as_final = (
        model_name == "fleet_updated_deterministic" and physically_valid
    )

    return {
        "target_battery_id": target_id,
        "model_name": model_name,
        "model_role": model_role,
        "n_cycles": int(len(full_table)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_fleet_batteries": int(n_fleet_batteries),
        "top_k_neighbors": int(top_k),
        "top_k_battery_ids": ",".join(top_ids),
        "train_rmse_soh": train_rmse,
        "train_mae_soh": train_mae,
        "test_rmse_soh": test_rmse,
        "test_mae_soh": test_mae,
        "first_test_jump_soh": first_test_jump,
        "n_prediction_increases": n_increases,
        "max_prediction_increase": max_increase,
        "physically_valid_prediction": bool(physically_valid),
        "selected_as_final": bool(selected_as_final),
    }


def fit_models_for_target(target_id, cycle_dataset, all_curves):
    target_table = cycle_dataset[cycle_dataset["battery_id"] == target_id].copy()
    train, test = split_target(target_table)
    full = pd.concat([train, test], ignore_index=True)
    split_labels = np.array(["train"] * len(train) + ["test"] * len(test))
    x_all = full["energy_frac"].to_numpy(dtype=float)
    x_train = train["energy_frac"].to_numpy(dtype=float)
    y_train = train["SOH"].to_numpy(dtype=float)

    fleet_curves = {
        battery_id: curve
        for battery_id, curve in all_curves.items()
        if battery_id != target_id
    }
    top_k = choose_k_by_train_validation(target_id, train, fleet_curves)
    components = build_components(target_id, train, fleet_curves, top_k)
    top_ids = components[components["selected_top_k"]]["fleet_battery_id"].tolist()

    fleet_prior_pred = weighted_prior(fleet_curves, components, x_all)
    residual_train = y_train - fleet_prior_pred[: len(train)]
    updated_pred = fleet_prior_pred + residual_update_curve(
        x_train, residual_train, x_all
    )
    updated_pred = enforce_final_prediction_rules(train, test, updated_pred)

    self_curve = MonotoneAgingCurve(x_train, y_train)
    self_pred = self_curve(x_all)
    self_pred = enforce_final_prediction_rules(train, test, self_pred)

    model_predictions = {
        "self_only_energy_baseline": (
            "baseline",
            self_pred,
        ),
        "fleet_prior_weighted_raw": (
            "diagnostic_prior",
            fleet_prior_pred,
        ),
        "fleet_updated_deterministic": (
            "final_model",
            updated_pred,
        ),
    }

    predictions = []
    metrics = []
    for model_name in MODEL_NAMES:
        model_role, pred = model_predictions[model_name]
        predictions.append(
            prediction_rows(target_id, full, split_labels, model_name, model_role, pred)
        )
        metrics.append(
            metric_row(
                target_id,
                model_name,
                model_role,
                full,
                split_labels,
                pred,
                len(fleet_curves),
                top_k,
                top_ids,
            )
        )

    return {
        "target_id": target_id,
        "train": train,
        "test": test,
        "full": full,
        "split_labels": split_labels,
        "x_all": x_all,
        "predictions": pd.concat(predictions, ignore_index=True),
        "metrics": pd.DataFrame(metrics),
        "components": components,
        "fleet_prior_pred": fleet_prior_pred,
        "self_pred": self_pred,
        "updated_pred": updated_pred,
        "top_ids": top_ids,
        "top_k": top_k,
        "fleet_curves": fleet_curves,
    }


def plot_fleet_update(results_by_target):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, target_id in zip(axes, TARGET_BATTERIES):
        result = results_by_target[target_id]
        train = result["train"]
        test = result["test"]
        x_grid = np.linspace(0.0, 1.0, 250)

        for index, curve in enumerate(result["fleet_curves"].values()):
            ax.plot(
                x_grid,
                curve(x_grid),
                color="0.78",
                linewidth=0.7,
                alpha=0.55,
                label="Individual fleet curves" if index == 0 else None,
            )

        ax.scatter(
            train["energy_frac"],
            train["SOH"],
            s=18,
            color="black",
            label="Target train SOH",
            zorder=5,
        )
        ax.scatter(
            test["energy_frac"],
            test["SOH"],
            s=26,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            label="Target test SOH",
            zorder=5,
        )
        ax.plot(
            result["x_all"],
            result["fleet_prior_pred"],
            color="#e68613",
            linestyle="--",
            linewidth=2.0,
            label="Weighted fleet prior",
        )
        ax.plot(
            result["x_all"],
            result["self_pred"],
            color="#2f6db3",
            linewidth=2.0,
            label="Self-only baseline",
        )
        ax.plot(
            result["x_all"],
            result["updated_pred"],
            color="#208744",
            linewidth=3.0,
            label="Fleet-updated deterministic model",
        )
        split_x = float(train["energy_frac"].iloc[-1])
        ax.axvline(split_x, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(target_id)
        ax.set_xlabel("Normalized cumulative energy age")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Deterministic fleet-prior update for capacity aging", fontsize=16)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(FLEET_UPDATE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def plot_model_comparison(results_by_target):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, target_id in zip(axes, TARGET_BATTERIES):
        result = results_by_target[target_id]
        full = result["full"]
        train = result["train"]
        test = result["test"]
        ax.plot(
            full["energy_frac"],
            full["SOH"],
            color="black",
            marker="o",
            markersize=3,
            linewidth=0.8,
            alpha=0.65,
            label="Measured raw SOH",
        )
        ax.plot(
            result["x_all"],
            result["self_pred"],
            color="#2f6db3",
            linewidth=2.0,
            label="Self-only baseline",
        )
        ax.plot(
            result["x_all"],
            result["updated_pred"],
            color="#208744",
            linewidth=3.0,
            label="Fleet-updated deterministic final model",
        )
        ax.axvline(
            float(train["energy_frac"].iloc[-1]),
            color="0.25",
            linestyle="--",
            linewidth=1.0,
            label="Train/test split",
        )
        ax.scatter(
            test["energy_frac"],
            test["SOH"],
            s=24,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
        )
        ax.set_title(target_id)
        ax.set_xlabel("Normalized cumulative energy age")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(MODEL_COMPARISON_FIGURE_PATH, dpi=200)
    plt.close(fig)


def plot_rmse_summary(metrics):
    summary = (
        metrics[metrics["model_name"].isin(["self_only_energy_baseline", "fleet_updated_deterministic"])]
        .groupby("model_name", as_index=False)["test_rmse_soh"]
        .mean()
    )
    labels = {
        "self_only_energy_baseline": "Self-only energy baseline",
        "fleet_updated_deterministic": "Fleet-updated deterministic",
    }
    colors = {
        "self_only_energy_baseline": "#2f6db3",
        "fleet_updated_deterministic": "#208744",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [labels[name] for name in summary["model_name"]],
        summary["test_rmse_soh"],
        color=[colors[name] for name in summary["model_name"]],
    )
    ax.set_ylabel("Average test RMSE (SOH)")
    ax.set_title("Capacity-aging v2 test RMSE summary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RMSE_SUMMARY_FIGURE_PATH, dpi=200)
    plt.close(fig)


def print_summary(cycle_dataset, results_by_target, metrics):
    print("\nExample 17 v2 results")
    print(f"Valid batteries used: {', '.join(sorted(cycle_dataset['battery_id'].unique()))}")
    print(f"Target batteries evaluated: {', '.join(TARGET_BATTERIES)}")

    average_rmse = metrics.groupby("model_name")["test_rmse_soh"].mean()
    print("\nAverage test RMSE per model:")
    for model_name, value in average_rmse.items():
        print(f"  {model_name}: {value:.5f}")

    print("\nPer-target test RMSE:")
    for row in metrics.itertuples(index=False):
        print(
            f"  {row.target_battery_id} | {row.model_name}: "
            f"{row.test_rmse_soh:.5f}"
        )

    print("\nTop-k neighbors:")
    for target_id in TARGET_BATTERIES:
        result = results_by_target[target_id]
        print(
            f"  {target_id}: k={result['top_k']} -> "
            f"{', '.join(result['top_ids'])}"
        )

    self_rmse = average_rmse["self_only_energy_baseline"]
    fleet_rmse = average_rmse["fleet_updated_deterministic"]
    if fleet_rmse < self_rmse:
        print(
            "\nfleet_updated_deterministic improves over "
            "self_only_energy_baseline on average test RMSE."
        )
    else:
        print(
            "\nfleet_updated_deterministic does not improve over "
            "self_only_energy_baseline on average test RMSE."
        )

    print(
        "\nTemperature is not used in v2 capacity-aging update because the previous "
        "audit showed no consistent capacity-aging improvement."
    )
    print("The final v2 model is fleet_updated_deterministic.")
    print(
        "Example 17 v2 is a deterministic fleet-prior update demo for cycle-level "
        "capacity aging. It is inspired by the fleet-prior idea in the reference "
        "paper, but it is not a Bayesian posterior update and does not implement "
        "qmax/R0, vMLP, uncertainty bands, or partial-observation inference."
    )


def warn_if_final_curves_nearly_linear(results_by_target):
    for target_id, result in results_by_target.items():
        x = result["x_all"]
        y = result["updated_pred"]
        linear = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
        max_deviation = float(np.max(np.abs(y - linear)))
        if max_deviation < 0.003:
            print(
                f"Warning: final learned curve for {target_id} is nearly linear "
                f"(max endpoint-line deviation {max_deviation:.5f})."
            )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {INPUT_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    sequence_table = pd.read_csv(INPUT_PATH)
    cycle_dataset, skipped = build_cycle_dataset(sequence_table)
    cycle_dataset.to_csv(CYCLE_DATASET_PATH, index=False)

    valid_batteries = sorted(cycle_dataset["battery_id"].unique())
    missing_targets = [target for target in TARGET_BATTERIES if target not in valid_batteries]
    if missing_targets:
        raise ValueError(f"Target batteries are not valid or missing: {missing_targets}")

    all_curves = {}
    for battery_id in valid_batteries:
        battery_table = cycle_dataset[cycle_dataset["battery_id"] == battery_id]
        all_curves[battery_id] = MonotoneAgingCurve(
            battery_table["energy_frac"], battery_table["SOH"]
        )

    results_by_target = {}
    for target_id in TARGET_BATTERIES:
        results_by_target[target_id] = fit_models_for_target(
            target_id, cycle_dataset, all_curves
        )

    predictions = pd.concat(
        [result["predictions"] for result in results_by_target.values()],
        ignore_index=True,
    )
    metrics = pd.concat(
        [result["metrics"] for result in results_by_target.values()],
        ignore_index=True,
    )
    components = pd.concat(
        [result["components"] for result in results_by_target.values()],
        ignore_index=True,
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    components.to_csv(FLEET_COMPONENTS_PATH, index=False)

    plot_fleet_update(results_by_target)
    plot_model_comparison(results_by_target)
    plot_rmse_summary(metrics)

    warn_if_final_curves_nearly_linear(results_by_target)
    print_summary(cycle_dataset, results_by_target, metrics)

    if skipped:
        print("\nSkipped batteries:")
        for battery_id, reason in skipped.items():
            print(f"  {battery_id}: {reason}")


if __name__ == "__main__":
    main()
