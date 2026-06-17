from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


REQUIRED_COLUMNS = [
    "battery_id",
    "discharge_index",
    "capacity_ah",
    "used_capacity_ah",
    "current_a",
    "voltage_v",
    "temperature_c",
    "dt_s",
    "cumulative_energy_kwh",
]

BASELINE_FEATURES = [
    "soc",
    "soc_squared",
    "soc_cubed",
    "discharge_current_a",
]

RESIDUAL_FEATURES = [
    "soc",
    "soc_squared",
    "soc_cubed",
    "discharge_current_a",
    "used_capacity_ah",
    "cumulative_energy_kwh",
    "discharge_index",
]

TEMPERATURE_RESIDUAL_FEATURES = RESIDUAL_FEATURES + ["temperature_c"]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def score_predictions(table, prediction_column):
    return {
        "rmse_v": rmse(table["voltage_v"], table[prediction_column]),
        "mae_v": mean_absolute_error(table["voltage_v"], table[prediction_column]),
    }


def split_by_cycle(table):
    all_cycles = sorted(table["discharge_index"].unique())
    split_index = int(0.7 * len(all_cycles))
    train_cycles = all_cycles[:split_index]
    test_cycles = all_cycles[split_index:]

    train_table = table[table["discharge_index"].isin(train_cycles)].copy()
    test_table = table[table["discharge_index"].isin(test_cycles)].copy()

    return all_cycles, train_cycles, test_cycles, train_table, test_table


def fit_battery_models(battery_id, battery_table):
    model_table = battery_table[battery_table["capacity_is_observed"]].copy()
    model_table = model_table.dropna(subset=REQUIRED_COLUMNS)

    if model_table.empty:
        return None, None, "no rows after observed-capacity and required-column filtering"

    if (model_table["capacity_ah"] <= 0.0).any():
        bad_rows = int((model_table["capacity_ah"] <= 0.0).sum())
        return None, None, (
            "non-positive capacity prevents finite SOC computation "
            f"({bad_rows} rows)"
        )

    model_table["soc"] = (
        1.0 - model_table["used_capacity_ah"] / model_table["capacity_ah"]
    )
    model_table["soc"] = model_table["soc"].clip(lower=0.0, upper=1.0)
    model_table["soc_squared"] = model_table["soc"] ** 2
    model_table["soc_cubed"] = model_table["soc"] ** 3
    model_table["discharge_current_a"] = (-model_table["current_a"]).clip(lower=0.0)

    (
        all_cycles,
        train_cycles,
        test_cycles,
        train_table,
        test_table,
    ) = split_by_cycle(model_table)

    if len(all_cycles) < 4:
        return None, None, f"too few discharge cycles after filtering: {len(all_cycles)}"

    if len(train_cycles) == 0 or len(test_cycles) == 0:
        return None, None, (
            "chronological split produced an empty train or test set "
            f"({len(train_cycles)} train cycles, {len(test_cycles)} test cycles)"
        )

    baseline_model = LinearRegression()
    baseline_model.fit(train_table[BASELINE_FEATURES], train_table["voltage_v"])

    train_table["voltage_baseline_pred_v"] = baseline_model.predict(
        train_table[BASELINE_FEATURES]
    )
    test_table["voltage_baseline_pred_v"] = baseline_model.predict(
        test_table[BASELINE_FEATURES]
    )

    train_table["residual_v"] = (
        train_table["voltage_v"] - train_table["voltage_baseline_pred_v"]
    )

    residual_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    residual_model.fit(train_table[RESIDUAL_FEATURES], train_table["residual_v"])

    temperature_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    temperature_model.fit(
        train_table[TEMPERATURE_RESIDUAL_FEATURES],
        train_table["residual_v"],
    )

    for split_table in [train_table, test_table]:
        split_table["residual_pred_no_temperature_v"] = residual_model.predict(
            split_table[RESIDUAL_FEATURES]
        )
        split_table["residual_pred_temperature_v"] = temperature_model.predict(
            split_table[TEMPERATURE_RESIDUAL_FEATURES]
        )
        split_table["voltage_hybrid_pred_v"] = (
            split_table["voltage_baseline_pred_v"]
            + split_table["residual_pred_no_temperature_v"]
        )
        split_table["voltage_temperature_hybrid_pred_v"] = (
            split_table["voltage_baseline_pred_v"]
            + split_table["residual_pred_temperature_v"]
        )

    baseline_train = score_predictions(train_table, "voltage_baseline_pred_v")
    baseline_test = score_predictions(test_table, "voltage_baseline_pred_v")
    hybrid_train = score_predictions(train_table, "voltage_hybrid_pred_v")
    hybrid_test = score_predictions(test_table, "voltage_hybrid_pred_v")
    temperature_train = score_predictions(
        train_table,
        "voltage_temperature_hybrid_pred_v",
    )
    temperature_test = score_predictions(
        test_table,
        "voltage_temperature_hybrid_pred_v",
    )

    metrics_rows = [
        {
            "battery_id": battery_id,
            "model_name": "physics_inspired_baseline",
            "number_of_discharge_cycles": len(all_cycles),
            "number_of_train_cycles": len(train_cycles),
            "number_of_test_cycles": len(test_cycles),
            "number_of_train_rows": len(train_table),
            "number_of_test_rows": len(test_table),
            "train_rmse_v": baseline_train["rmse_v"],
            "test_rmse_v": baseline_test["rmse_v"],
            "train_mae_v": baseline_train["mae_v"],
            "test_mae_v": baseline_test["mae_v"],
            "baseline_intercept": baseline_model.intercept_,
            "coef_soc": baseline_model.coef_[0],
            "coef_soc_squared": baseline_model.coef_[1],
            "coef_soc_cubed": baseline_model.coef_[2],
            "coef_discharge_current_a": baseline_model.coef_[3],
        },
        {
            "battery_id": battery_id,
            "model_name": "hybrid_residual_without_temperature",
            "number_of_discharge_cycles": len(all_cycles),
            "number_of_train_cycles": len(train_cycles),
            "number_of_test_cycles": len(test_cycles),
            "number_of_train_rows": len(train_table),
            "number_of_test_rows": len(test_table),
            "train_rmse_v": hybrid_train["rmse_v"],
            "test_rmse_v": hybrid_test["rmse_v"],
            "train_mae_v": hybrid_train["mae_v"],
            "test_mae_v": hybrid_test["mae_v"],
            "baseline_intercept": baseline_model.intercept_,
            "coef_soc": baseline_model.coef_[0],
            "coef_soc_squared": baseline_model.coef_[1],
            "coef_soc_cubed": baseline_model.coef_[2],
            "coef_discharge_current_a": baseline_model.coef_[3],
        },
        {
            "battery_id": battery_id,
            "model_name": "temperature_aware_hybrid_residual",
            "number_of_discharge_cycles": len(all_cycles),
            "number_of_train_cycles": len(train_cycles),
            "number_of_test_cycles": len(test_cycles),
            "number_of_train_rows": len(train_table),
            "number_of_test_rows": len(test_table),
            "train_rmse_v": temperature_train["rmse_v"],
            "test_rmse_v": temperature_test["rmse_v"],
            "train_mae_v": temperature_train["mae_v"],
            "test_mae_v": temperature_test["mae_v"],
            "baseline_intercept": baseline_model.intercept_,
            "coef_soc": baseline_model.coef_[0],
            "coef_soc_squared": baseline_model.coef_[1],
            "coef_soc_cubed": baseline_model.coef_[2],
            "coef_discharge_current_a": baseline_model.coef_[3],
        },
    ]

    per_cycle_rows = []

    for discharge_index, cycle_table in test_table.groupby("discharge_index"):
        baseline_cycle = score_predictions(cycle_table, "voltage_baseline_pred_v")
        hybrid_cycle = score_predictions(cycle_table, "voltage_hybrid_pred_v")
        temperature_cycle = score_predictions(
            cycle_table,
            "voltage_temperature_hybrid_pred_v",
        )

        per_cycle_rows.append(
            {
                "battery_id": battery_id,
                "discharge_index": discharge_index,
                "number_of_steps": len(cycle_table),
                "baseline_rmse_v": baseline_cycle["rmse_v"],
                "hybrid_rmse_v": hybrid_cycle["rmse_v"],
                "temperature_hybrid_rmse_v": temperature_cycle["rmse_v"],
                "baseline_mae_v": baseline_cycle["mae_v"],
                "hybrid_mae_v": hybrid_cycle["mae_v"],
                "temperature_hybrid_mae_v": temperature_cycle["mae_v"],
                "hybrid_rmse_improvement_v": (
                    baseline_cycle["rmse_v"] - hybrid_cycle["rmse_v"]
                ),
                "temperature_rmse_gain_v": (
                    hybrid_cycle["rmse_v"] - temperature_cycle["rmse_v"]
                ),
                "hybrid_mae_improvement_v": (
                    baseline_cycle["mae_v"] - hybrid_cycle["mae_v"]
                ),
                "temperature_mae_gain_v": (
                    hybrid_cycle["mae_v"] - temperature_cycle["mae_v"]
                ),
            }
        )

    return metrics_rows, per_cycle_rows, None


def plot_rmse_summary(metrics_table, figure_path):
    summary = metrics_table.pivot(
        index="battery_id",
        columns="model_name",
        values="test_rmse_v",
    ).sort_index()

    labels = {
        "physics_inspired_baseline": "Baseline",
        "hybrid_residual_without_temperature": "Hybrid",
        "temperature_aware_hybrid_residual": "Temp hybrid",
    }

    x = np.arange(len(summary.index))
    width = 0.25

    plt.figure(figsize=(10, 5))

    for offset, model_name in zip([-width, 0, width], labels):
        plt.bar(
            x + offset,
            summary[model_name],
            width=width,
            label=labels[model_name],
        )

    plt.xticks(x, summary.index, rotation=45)
    plt.ylabel("Test RMSE [V]")
    plt.title("Multi-battery voltage response test RMSE")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def plot_temperature_gain(per_cycle_metrics, figure_path):
    gain_by_battery = (
        per_cycle_metrics.groupby("battery_id")["temperature_rmse_gain_v"]
        .mean()
        .sort_index()
    )

    colors = np.where(gain_by_battery >= 0, "tab:green", "tab:red")

    plt.figure(figsize=(8, 5))
    plt.bar(gain_by_battery.index, gain_by_battery.values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Mean test-cycle RMSE gain [V]")
    plt.title("Temperature-aware residual gain by battery")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parents[1]
    sequence_path = project_root / "results" / "discharge_sequences_all.csv"
    output_dir = project_root / "results"
    figure_dir = project_root / "figures"

    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    metrics_path = output_dir / "multi_battery_voltage_response_metrics.csv"
    per_cycle_metrics_path = (
        output_dir / "multi_battery_voltage_response_per_cycle_metrics.csv"
    )
    rmse_figure_path = figure_dir / "multi_battery_voltage_rmse_summary.png"
    temperature_gain_figure_path = figure_dir / "temperature_gain_multi_battery.png"

    table = pd.read_csv(sequence_path)

    metrics_rows = []
    per_cycle_rows = []
    skipped_rows = []

    for battery_id in sorted(table["battery_id"].dropna().unique()):
        battery_table = table[table["battery_id"] == battery_id].copy()
        metrics, per_cycle_metrics, skip_reason = fit_battery_models(
            battery_id,
            battery_table,
        )

        if skip_reason is not None:
            skipped_rows.append(
                {
                    "battery_id": battery_id,
                    "skip_reason": skip_reason,
                }
            )
            continue

        metrics_rows.extend(metrics)
        per_cycle_rows.extend(per_cycle_metrics)

    metrics_table = pd.DataFrame(metrics_rows)
    per_cycle_metrics = pd.DataFrame(per_cycle_rows)
    skipped_table = pd.DataFrame(skipped_rows)

    metrics_table.to_csv(metrics_path, index=False)
    per_cycle_metrics.to_csv(per_cycle_metrics_path, index=False)

    if not metrics_table.empty:
        plot_rmse_summary(metrics_table, rmse_figure_path)

    if not per_cycle_metrics.empty:
        plot_temperature_gain(per_cycle_metrics, temperature_gain_figure_path)

    if metrics_table.empty:
        print("No batteries were used.")
        return

    metrics_wide = metrics_table.pivot(
        index="battery_id",
        columns="model_name",
        values="test_rmse_v",
    )

    baseline_rmse = metrics_wide["physics_inspired_baseline"]
    hybrid_rmse = metrics_wide["hybrid_residual_without_temperature"]
    temperature_rmse = metrics_wide["temperature_aware_hybrid_residual"]

    hybrid_improved = (hybrid_rmse < baseline_rmse).sum()
    temperature_improved = (temperature_rmse < hybrid_rmse).sum()

    print("Multi-battery voltage-response summary")
    print("--------------------------------------")
    print(f"Number of batteries used: {len(metrics_wide)}")

    print("\nSkipped batteries and reasons:")
    if skipped_table.empty:
        print("None")
    else:
        for row in skipped_table.itertuples(index=False):
            print(f"{row.battery_id}: {row.skip_reason}")

    print(f"\nAverage baseline test RMSE: {baseline_rmse.mean():.6f} V")
    print(f"Average hybrid test RMSE: {hybrid_rmse.mean():.6f} V")
    print(
        "Average temperature-aware hybrid test RMSE: "
        f"{temperature_rmse.mean():.6f} V"
    )
    print(
        "Number of batteries improved by hybrid residual learning: "
        f"{hybrid_improved}"
    )
    print(
        "Number of batteries improved by temperature-aware residual learning: "
        f"{temperature_improved}"
    )

    print("\nOutput file paths:")
    print(metrics_path)
    print(per_cycle_metrics_path)
    print(rmse_figure_path)
    print(temperature_gain_figure_path)


if __name__ == "__main__":
    main()
