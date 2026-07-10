from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "example_19_final_summary"

VOLTAGE_SUMMARY_PATH = OUTPUT_DIR / "final_voltage_summary.csv"
CAPACITY_SUMMARY_PATH = OUTPUT_DIR / "final_capacity_summary.csv"
OVERVIEW_SUMMARY_PATH = OUTPUT_DIR / "final_overview_summary.csv"
VOLTAGE_FIGURE_PATH = OUTPUT_DIR / "final_voltage_comparison.png"
CAPACITY_FIGURE_PATH = OUTPUT_DIR / "final_capacity_comparison.png"

EXPECTED_INPUTS = {
    "multi_battery_voltage": RESULTS_DIR / "multi_battery_voltage_response_metrics.csv",
    "temperature_ablation": RESULTS_DIR / "temperature_ablation_comparison_B0005.csv",
    "capacity_aging_final": RESULTS_DIR / "capacity_aging_final_metrics.csv",
    "observation_update": RESULTS_DIR / "capacity_aging_observation_update_metrics.csv",
}


def read_csv_if_available(label, path, warnings, found):
    if not path.exists():
        warnings.append(f"Missing {label}: {path.relative_to(PROJECT_ROOT)}")
        return None
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Could not read {label}: {path.relative_to(PROJECT_ROOT)} ({exc})")
        return None
    found.append(label)
    return table


def mean_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def median_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def summarize_voltage(warnings, found):
    rows = []

    multi = read_csv_if_available(
        "multi_battery_voltage",
        EXPECTED_INPUTS["multi_battery_voltage"],
        warnings,
        found,
    )
    if multi is not None:
        required = {"model_name", "battery_id", "test_rmse_v", "test_mae_v"}
        if required.issubset(multi.columns):
            for model_name, group in multi.groupby("model_name", sort=True):
                rows.append(
                    {
                        "experiment": "multi_battery_voltage_response",
                        "source_file": EXPECTED_INPUTS[
                            "multi_battery_voltage"
                        ].relative_to(PROJECT_ROOT).as_posix(),
                        "model_name": model_name,
                        "n_batteries": int(group["battery_id"].nunique()),
                        "n_rows": int(len(group)),
                        "mean_test_rmse_v": mean_or_nan(group["test_rmse_v"]),
                        "median_test_rmse_v": median_or_nan(group["test_rmse_v"]),
                        "mean_test_mae_v": mean_or_nan(group["test_mae_v"]),
                        "mean_rmse_improvement_vs_baseline_percent": np.nan,
                        "notes": "multi-battery held-out voltage prediction",
                    }
                )
        else:
            warnings.append("multi_battery_voltage file exists but lacks required columns")

    ablation = read_csv_if_available(
        "temperature_ablation",
        EXPECTED_INPUTS["temperature_ablation"],
        warnings,
        found,
    )
    if ablation is not None:
        required = {
            "no_temperature_rmse_v",
            "no_temperature_mae_v",
            "temperature_rmse_v",
            "temperature_mae_v",
            "temperature_improved_rmse",
        }
        if required.issubset(ablation.columns):
            no_temp_rmse = mean_or_nan(ablation["no_temperature_rmse_v"])
            temp_rmse = mean_or_nan(ablation["temperature_rmse_v"])
            improvement = (
                100.0 * (no_temp_rmse - temp_rmse) / no_temp_rmse
                if np.isfinite(no_temp_rmse) and no_temp_rmse > 0.0
                else np.nan
            )
            rows.extend(
                [
                    {
                        "experiment": "temperature_ablation_B0005",
                        "source_file": EXPECTED_INPUTS[
                            "temperature_ablation"
                        ].relative_to(PROJECT_ROOT).as_posix(),
                        "model_name": "hybrid_residual_without_temperature",
                        "n_batteries": 1,
                        "n_rows": int(len(ablation)),
                        "mean_test_rmse_v": no_temp_rmse,
                        "median_test_rmse_v": median_or_nan(
                            ablation["no_temperature_rmse_v"]
                        ),
                        "mean_test_mae_v": mean_or_nan(
                            ablation["no_temperature_mae_v"]
                        ),
                        "mean_rmse_improvement_vs_baseline_percent": 0.0,
                        "notes": "B0005 per-cycle temperature ablation baseline",
                    },
                    {
                        "experiment": "temperature_ablation_B0005",
                        "source_file": EXPECTED_INPUTS[
                            "temperature_ablation"
                        ].relative_to(PROJECT_ROOT).as_posix(),
                        "model_name": "temperature_aware_hybrid_residual",
                        "n_batteries": 1,
                        "n_rows": int(len(ablation)),
                        "mean_test_rmse_v": temp_rmse,
                        "median_test_rmse_v": median_or_nan(
                            ablation["temperature_rmse_v"]
                        ),
                        "mean_test_mae_v": mean_or_nan(ablation["temperature_mae_v"]),
                        "mean_rmse_improvement_vs_baseline_percent": improvement,
                        "notes": (
                            "B0005 per-cycle temperature ablation; "
                            f"improved {int(ablation['temperature_improved_rmse'].sum())}/"
                            f"{len(ablation)} cycles by RMSE"
                        ),
                    },
                ]
            )
        else:
            warnings.append("temperature_ablation file exists but lacks required columns")

    voltage = pd.DataFrame(rows)
    if not voltage.empty and "multi_battery_voltage_response" in voltage["experiment"].values:
        baseline = voltage[
            (voltage["experiment"] == "multi_battery_voltage_response")
            & (voltage["model_name"] == "physics_inspired_baseline")
        ]
        if not baseline.empty:
            baseline_rmse = float(baseline["mean_test_rmse_v"].iloc[0])
            mask = voltage["experiment"] == "multi_battery_voltage_response"
            voltage.loc[mask, "mean_rmse_improvement_vs_baseline_percent"] = (
                100.0
                * (baseline_rmse - voltage.loc[mask, "mean_test_rmse_v"])
                / baseline_rmse
            )
    return voltage


def summarize_capacity(warnings, found):
    rows = []

    final17 = read_csv_if_available(
        "capacity_aging_final",
        EXPECTED_INPUTS["capacity_aging_final"],
        warnings,
        found,
    )
    if final17 is not None:
        required = {"battery_id", "model_name", "model_role", "test_rmse_soh", "test_mae_soh"}
        if required.issubset(final17.columns):
            for keys, group in final17.groupby(["model_name", "model_role"], sort=True):
                model_name, model_role = keys
                rows.append(
                    {
                        "experiment": "example17_capacity_aging_final",
                        "source_file": EXPECTED_INPUTS[
                            "capacity_aging_final"
                        ].relative_to(PROJECT_ROOT).as_posix(),
                        "case_or_model": model_name,
                        "model_role": model_role,
                        "n_batteries": int(group["battery_id"].nunique()),
                        "mean_rmse_soh": mean_or_nan(group["test_rmse_soh"]),
                        "median_rmse_soh": median_or_nan(group["test_rmse_soh"]),
                        "mean_mae_soh": mean_or_nan(group["test_mae_soh"]),
                        "mean_improvement_vs_reference_percent": mean_or_nan(
                            group["relative_improvement_vs_self_percent"]
                        )
                        if "relative_improvement_vs_self_percent" in group
                        else np.nan,
                        "notes": "final 70/30 capacity-aging forecast",
                    }
                )
        else:
            warnings.append("capacity_aging_final file exists but lacks required columns")

    example18 = read_csv_if_available(
        "observation_update",
        EXPECTED_INPUTS["observation_update"],
        warnings,
        found,
    )
    if example18 is not None:
        required = {"battery_id", "case", "rmse_soh", "mae_soh"}
        if required.issubset(example18.columns):
            for case, group in example18.groupby("case", sort=True):
                rows.append(
                    {
                        "experiment": "example18_observation_update",
                        "source_file": EXPECTED_INPUTS[
                            "observation_update"
                        ].relative_to(PROJECT_ROOT).as_posix(),
                        "case_or_model": case,
                        "model_role": "observation_update_case",
                        "n_batteries": int(group["battery_id"].nunique()),
                        "mean_rmse_soh": mean_or_nan(group["rmse_soh"]),
                        "median_rmse_soh": median_or_nan(group["rmse_soh"]),
                        "mean_mae_soh": mean_or_nan(group["mae_soh"]),
                        "mean_improvement_vs_reference_percent": mean_or_nan(
                            group["improvement_vs_no_update_percent"]
                        )
                        if "improvement_vs_no_update_percent" in group
                        else np.nan,
                        "notes": "Example 18 evaluation on normalized energy age x > 0.80",
                    }
                )
        else:
            warnings.append("observation_update file exists but lacks required columns")

    return pd.DataFrame(rows)


def build_overview(voltage, capacity):
    rows = []
    if not voltage.empty:
        for experiment, group in voltage.groupby("experiment", sort=True):
            ranked = group.dropna(subset=["mean_test_rmse_v"]).sort_values(
                "mean_test_rmse_v"
            )
            if ranked.empty:
                continue
            best = ranked.iloc[0]
            rows.append(
                {
                    "domain": "voltage_prediction",
                    "experiment": experiment,
                    "best_summary_label": f"{best['experiment']}::{best['model_name']}",
                    "primary_metric": "mean_test_rmse_v",
                    "primary_metric_value": float(best["mean_test_rmse_v"]),
                    "n_summary_rows": int(len(group)),
                    "notes": "lower voltage RMSE is better",
                }
            )

    if not capacity.empty:
        for experiment, group in capacity.groupby("experiment", sort=True):
            ranked = group.dropna(subset=["mean_rmse_soh"]).sort_values(
                "mean_rmse_soh"
            )
            if ranked.empty:
                continue
            best = ranked.iloc[0]
            rows.append(
                {
                    "domain": "capacity_aging",
                    "experiment": experiment,
                    "best_summary_label": f"{best['experiment']}::{best['case_or_model']}",
                    "primary_metric": "mean_rmse_soh",
                    "primary_metric_value": float(best["mean_rmse_soh"]),
                    "n_summary_rows": int(len(group)),
                    "notes": "lower SOH RMSE is better",
                }
            )
    return pd.DataFrame(rows)


def plot_bar_summary(table, value_column, label_column, title, ylabel, path):
    if table.empty or value_column not in table:
        return False
    plot_table = table.dropna(subset=[value_column]).copy()
    if plot_table.empty:
        return False
    plot_table["plot_label"] = (
        plot_table["experiment"].astype(str) + "\n" + plot_table[label_column].astype(str)
    )
    plot_table = plot_table.sort_values(value_column)

    fig_height = max(4.8, 0.42 * len(plot_table) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y = np.arange(len(plot_table))
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, len(plot_table)))
    ax.barh(y, plot_table[value_column].to_numpy(dtype=float), color=colors)
    ax.set_yticks(y, plot_table["plot_label"])
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(plot_table[value_column].to_numpy(dtype=float)):
        ax.text(value, index, f" {value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings = []
    found = []
    generated = []

    voltage = summarize_voltage(warnings, found)
    capacity = summarize_capacity(warnings, found)
    overview = build_overview(voltage, capacity)

    voltage.to_csv(VOLTAGE_SUMMARY_PATH, index=False)
    capacity.to_csv(CAPACITY_SUMMARY_PATH, index=False)
    overview.to_csv(OVERVIEW_SUMMARY_PATH, index=False)
    generated.extend([VOLTAGE_SUMMARY_PATH, CAPACITY_SUMMARY_PATH, OVERVIEW_SUMMARY_PATH])

    if plot_bar_summary(
        voltage,
        "mean_test_rmse_v",
        "model_name",
        "Final Voltage Prediction Summary",
        "Mean test RMSE (V)",
        VOLTAGE_FIGURE_PATH,
    ):
        generated.append(VOLTAGE_FIGURE_PATH)
    else:
        warnings.append("Skipped voltage comparison plot because no voltage summary rows were available")

    if plot_bar_summary(
        capacity,
        "mean_rmse_soh",
        "case_or_model",
        "Final Capacity Aging Summary",
        "Mean RMSE (SOH)",
        CAPACITY_FIGURE_PATH,
    ):
        generated.append(CAPACITY_FIGURE_PATH)
    else:
        warnings.append("Skipped capacity comparison plot because no capacity summary rows were available")

    print("\nExample 19 final summary")
    print("Experiments found:")
    if found:
        for label in found:
            print(f"  {label}")
    else:
        print("  none")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")

    print("\nGenerated summaries:")
    for path in generated:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    print(f"\nSaved outputs in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
