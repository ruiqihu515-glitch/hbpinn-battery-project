from pathlib import Path
import importlib.util
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
INPUT_PATH = RESULTS_DIR / "discharge_sequences_all.csv"

METRICS_PATH = RESULTS_DIR / "capacity_aging_observation_update_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "capacity_aging_observation_update_predictions.csv"
EXAMPLES_FIGURE_PATH = FIGURES_DIR / "capacity_aging_observation_update_examples.png"
RMSE_SUMMARY_FIGURE_PATH = FIGURES_DIR / "capacity_aging_observation_update_rmse_summary.png"
PER_BATTERY_RMSE_FIGURE_PATH = (
    FIGURES_DIR / "capacity_aging_observation_update_per_battery_rmse.png"
)
REPORT_PATH = PROJECT_ROOT / "capacity_aging_observation_update_report.md"

TARGET_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
SPLIT_X = 0.70
FULL_WINDOW_CUTOFF_X = 0.80
SPARSE_UPDATE_COUNT = 4
SOH_RANGE = (0.4, 1.05)


def load_example17_module():
    module_path = PROJECT_ROOT / "examples" / "17_capacity_aging_final.py"
    spec = importlib.util.spec_from_file_location("capacity_aging_final", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example17 = load_example17_module()


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def split_by_energy_fraction(table):
    table = table.sort_values("energy_frac").reset_index(drop=True)
    history = table[table["energy_frac"] <= SPLIT_X].copy()
    future = table[table["energy_frac"] > SPLIT_X].copy()
    if len(history) < 12 or len(future) < 6:
        raise ValueError("Not enough history/future points after the 0.70 split.")
    return history, future


def build_final17_forecast(cycle_dataset, battery_id):
    result = example17.evaluate_target(cycle_dataset, battery_id)
    return result["fleet_pred"]


def choose_sparse_update_points(update_window):
    early = update_window.copy()
    positions = np.linspace(0, len(early) - 1, min(SPARSE_UPDATE_COUNT, len(early)))
    indices = sorted(set(int(round(position)) for position in positions))
    return early.iloc[indices].copy()


def fit_residual_correction(update_x, update_y, base_update_pred, cutoff_x):
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


def metric_row(battery_id, case, update_start_x, update_end_x, history, update_points, eval_points, pred):
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


def add_matched_no_update_improvement(rows, full, base_pred):
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


def evaluate_battery(cycle_dataset, battery_id):
    full = (
        cycle_dataset[cycle_dataset["battery_id"] == battery_id]
        .sort_values("energy_frac")
        .reset_index(drop=True)
    )
    history, future = split_by_energy_fraction(full)
    x_full = full["energy_frac"].to_numpy(dtype=float)
    base_pred = build_final17_forecast(cycle_dataset, battery_id)

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
        metric_row(
            battery_id,
            "no_update",
            np.nan,
            FULL_WINDOW_CUTOFF_X,
            history,
            pd.DataFrame(),
            evaluation,
            base_pred,
        ),
        metric_row(
            battery_id,
            "partial_sparse_update",
            SPLIT_X,
            FULL_WINDOW_CUTOFF_X,
            history,
            sparse_update,
            evaluation,
            sparse_pred,
        ),
        metric_row(
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
    rows = add_matched_no_update_improvement(rows, full, base_pred)

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


def plot_observation_examples(results):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, battery_id in zip(axes, TARGET_BATTERIES):
        result = results[battery_id]
        full = result["full"]
        x = full["energy_frac"]
        ax.plot(
            x,
            full["SOH"],
            color="black",
            marker="o",
            markersize=3,
            linewidth=0.9,
            label="Measured raw SOH",
        )
        ax.plot(x, result["pred_no_update"], color="#2f6db3", linewidth=2.0, label="No update")
        ax.plot(
            x,
            result["pred_sparse"],
            color="#208744",
            linewidth=2.2,
            label="Partial sparse update",
        )
        ax.plot(
            x,
            result["pred_full_window"],
            color="#d95f02",
            linewidth=2.2,
            label="Full-window update",
        )
        ax.scatter(
            result["sparse_update"]["energy_frac"],
            result["sparse_update"]["SOH"],
            s=42,
            facecolors="white",
            edgecolors="#208744",
            linewidths=1.6,
            zorder=5,
            label="Sparse update points",
        )
        ax.axvline(SPLIT_X, color="0.25", linestyle="--", linewidth=1.1, label="0.70 split")
        ax.axvline(
            FULL_WINDOW_CUTOFF_X,
            color="0.45",
            linestyle="--",
            linewidth=1.1,
            label="0.80 update cutoff",
        )
        ax.set_title(battery_id)
        ax.set_xlabel("Normalized cumulative energy age")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Observation-update examples for capacity aging", fontsize=15)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig(EXAMPLES_FIGURE_PATH, dpi=200)
    plt.close(fig)


def plot_rmse_summary(metrics):
    order = ["no_update", "partial_sparse_update", "full_window_update"]
    labels = ["No update", "Partial sparse", "Full window"]
    colors = ["#2f6db3", "#208744", "#d95f02"]
    summary = metrics.groupby("case")["rmse_soh"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    bars = ax.bar(np.arange(len(order)), summary.to_numpy(), color=colors)
    ax.set_xticks(np.arange(len(order)), labels)
    ax.set_ylabel("Mean evaluation RMSE (SOH)")
    ax.set_title("Observation-update RMSE summary")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(RMSE_SUMMARY_FIGURE_PATH, dpi=200)
    plt.close(fig)


def plot_per_battery_rmse(metrics):
    order = ["no_update", "partial_sparse_update", "full_window_update"]
    labels = ["No update", "Partial sparse", "Full window"]
    colors = ["#2f6db3", "#208744", "#d95f02"]
    pivot = metrics.pivot(index="battery_id", columns="case", values="rmse_soh").reindex(TARGET_BATTERIES)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(TARGET_BATTERIES))
    width = 0.24
    offsets = [-width, 0.0, width]
    for offset, case, label, color in zip(offsets, order, labels, colors):
        ax.bar(x + offset, pivot[case].to_numpy(), width=width, label=label, color=color)
    ax.set_xticks(x, TARGET_BATTERIES)
    ax.set_ylabel("Evaluation RMSE (SOH)")
    ax.set_title("Observation-update RMSE by battery")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PER_BATTERY_RMSE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_report(metrics):
    mean_rmse = metrics.groupby("case")["rmse_soh"].mean()
    pivot = metrics.pivot(index="battery_id", columns="case", values="improvement_vs_no_update_percent")
    rmse_pivot = metrics.pivot(index="battery_id", columns="case", values="rmse_soh")

    report = f"""# Example 18: Full / Partial Observation Experiments

Example 18 completes the planned full / partial observation experiment as an exploratory continuation of final Example 17. It starts from the final Example 17 capacity-aging forecast for the same target batteries, then asks whether additional target-battery SOH observations can correct the future long-horizon forecast. It uses the same cleaned SOH/capacity pipeline, normalized cumulative-energy age definition, and 70% train/test split as final Example 17.

The base forecast is frozen after the Example 17 fit from the first 70% of target normalized energy age. The `no_update` case continues that final Example 17 forecast without using any target observations after x = 0.70. The `partial_sparse_update` case observes only {SPARSE_UPDATE_COUNT} sparse target points in 0.70 < x <= 0.80. The `full_window_update` case observes all target SOH points in 0.70 < x <= 0.80. All metrics are computed only on the future evaluation region x > 0.80.

This is not a full Bayesian posterior update. There is no posterior distribution, sampling, uncertainty propagation, qmax/R0 latent-state inference, or likelihood update. The update is deterministic and reproducible: it fits a small ridge-regularized residual correction, offset plus local slope, between the observed update SOH and the frozen no-update forecast.

Data leakage is avoided by construction. For each target battery, the initial forecast is the final Example 17 forecast trained from target history at x <= 0.70. The update cases use target observations only from 0.70 < x <= 0.80, and evaluation excludes the history and update windows. No SOH labels after x = 0.80 are used for fitting, tuning, smoothing, shifting, or calibration.

Mean RMSE by case:

| case | mean RMSE (SOH) |
| --- | ---: |
| no_update | {mean_rmse['no_update']:.5f} |
| partial_sparse_update | {mean_rmse['partial_sparse_update']:.5f} |
| full_window_update | {mean_rmse['full_window_update']:.5f} |

Per-battery improvement versus no update:

| battery | no update RMSE | partial RMSE | full RMSE | partial improvement | full improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for battery_id in TARGET_BATTERIES:
        report += (
            f"| {battery_id} | "
            f"{rmse_pivot.loc[battery_id, 'no_update']:.5f} | "
            f"{rmse_pivot.loc[battery_id, 'partial_sparse_update']:.5f} | "
            f"{rmse_pivot.loc[battery_id, 'full_window_update']:.5f} | "
            f"{pivot.loc[battery_id, 'partial_sparse_update']:.2f}% | "
            f"{pivot.loc[battery_id, 'full_window_update']:.2f}% |\n"
        )

    report += f"""
The best empirical mean result is the sparse partial update: mean RMSE improves from about {mean_rmse['no_update']:.4f} for `no_update` to about {mean_rmse['partial_sparse_update']:.4f} for `partial_sparse_update`. The full-window update is also slightly better than no update on average, at about {mean_rmse['full_window_update']:.4f}, but it is not better than the sparse update. The improvement is modest and battery-dependent, so this should not be presented as a decisive success.

B0018 is the main limitation: both observation updates worsen the future forecast there, showing that deterministic updates remain sensitive to late-life battery-specific behavior and local recovery/non-stationarity. The full-window update is not necessarily better than sparse observation because fitting more local noisy behavior in the update window can hurt future extrapolation after x = 0.80.

For the final report discussion, Example 18 should be described as an exploratory but logically complete full / partial observation experiment extending final Example 17. Sparse partial observations can improve aging forecasts on average, but deterministic updates remain sensitive to late-life battery-specific behavior and should not be framed as a decisive capacity-aging success.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def print_summary(metrics, generated_files):
    print("\nExample 18 deterministic observation-update experiment")
    print("Generated files:")
    for path in generated_files:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    print("\nMean RMSE per case:")
    for case, value in metrics.groupby("case")["rmse_soh"].mean().items():
        print(f"  {case}: {value:.5f}")

    print("\nPer-battery improvement vs no update:")
    pivot = metrics.pivot(index="battery_id", columns="case", values="improvement_vs_no_update_percent")
    for battery_id in TARGET_BATTERIES:
        print(
            f"  {battery_id}: partial_sparse_update="
            f"{pivot.loc[battery_id, 'partial_sparse_update']:+.2f}%, "
            f"full_window_update={pivot.loc[battery_id, 'full_window_update']:+.2f}%"
        )

    print(
        "\nConclusion: partial_sparse_update is the best empirical mean result, "
        "but the gain is modest and battery-dependent. B0018 remains the main "
        "limitation because both observation updates worsen the future forecast."
    )

    print("\ngit status --short:")
    status = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
    )
    print(status.stdout.rstrip() or "  clean")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input file not found: {INPUT_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    sequence_table = pd.read_csv(INPUT_PATH)
    cycle_dataset, skipped = example17.build_cycle_dataset(sequence_table)
    valid_batteries = set(cycle_dataset["battery_id"].unique())
    missing = [battery_id for battery_id in TARGET_BATTERIES if battery_id not in valid_batteries]
    if missing:
        raise ValueError(f"Target batteries are missing from the cleaned dataset: {missing}")

    results = {
        battery_id: evaluate_battery(cycle_dataset, battery_id)
        for battery_id in TARGET_BATTERIES
    }
    metrics = pd.concat([result["metrics"] for result in results.values()], ignore_index=True)
    predictions = pd.concat(
        [result["predictions"] for result in results.values()],
        ignore_index=True,
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    plot_observation_examples(results)
    plot_rmse_summary(metrics)
    plot_per_battery_rmse(metrics)
    write_report(metrics)

    generated_files = [
        METRICS_PATH,
        PREDICTIONS_PATH,
        EXAMPLES_FIGURE_PATH,
        RMSE_SUMMARY_FIGURE_PATH,
        PER_BATTERY_RMSE_FIGURE_PATH,
        REPORT_PATH,
    ]
    if skipped:
        print("Skipped non-target batteries during cleaning:")
        for battery_id, reason in skipped.items():
            print(f"  {battery_id}: {reason}")
    print_summary(metrics, generated_files)


if __name__ == "__main__":
    main()
