from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def main():
    project_root = Path(__file__).resolve().parents[1]
    battery_id = "B0005"
    prediction_path = project_root / "results" / f"physics_only_voltage_predictions_{battery_id}.csv"

    table = pd.read_csv(prediction_path)

    print("Loaded physics-inspired baseline prediction file:")
    print(prediction_path)

    print("\nTable shape:")
    print(table.shape)

    print("\nColumns:")
    print(table.columns.tolist())

    print("\nFirst rows:")
    print(table.head())

    table["residual_v"] = table["voltage_v"] - table["voltage_pred_v"]

    print("\nResidual range:")
    print(table["residual_v"].min(), table["residual_v"].max())

    print("\nResidual preview:")
    print(
        table[
            [
                "battery_id",
                "discharge_index",
                "step_index",
                "split",
                "voltage_v",
                "voltage_pred_v",
                "residual_v",
            ]
        ].head()
    )

    train_table = table[table["split"] == "train"].copy()
    test_table = table[table["split"] == "test"].copy()

    feature_columns = [
        "soc",
        "soc_squared",
        "soc_cubed",
        "discharge_current_a",
        "used_capacity_ah",
        "cumulative_energy_kwh",
        "discharge_index",
    ]

    X_train = train_table[feature_columns]
    y_train = train_table["residual_v"]

    X_test = test_table[feature_columns]
    y_test = test_table["residual_v"]

    print("\nResidual model feature columns:")
    print(feature_columns)

    print("\nTrain shape:")
    print(X_train.shape, y_train.shape)

    print("\nTest shape:")
    print(X_test.shape, y_test.shape)

    residual_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    residual_model.fit(X_train, y_train)

    train_residual_pred = residual_model.predict(X_train)
    test_residual_pred = residual_model.predict(X_test)

    print("\nResidual model fitted.")

    print("\nResidual prediction shape:")
    print(train_residual_pred.shape, test_residual_pred.shape)

    train_table["residual_pred_v"] = train_residual_pred
    test_table["residual_pred_v"] = test_residual_pred

    train_table["voltage_hybrid_pred_v"] = (
        train_table["voltage_pred_v"] + train_table["residual_pred_v"]
    )
    test_table["voltage_hybrid_pred_v"] = (
        test_table["voltage_pred_v"] + test_table["residual_pred_v"]
    )

    train_table["hybrid_error_v"] = (
        train_table["voltage_hybrid_pred_v"] - train_table["voltage_v"]
    )
    test_table["hybrid_error_v"] = (
        test_table["voltage_hybrid_pred_v"] - test_table["voltage_v"]
    )

    print("\nHybrid prediction preview:")
    print(
        test_table[
            [
                "battery_id",
                "discharge_index",
                "step_index",
                "voltage_v",
                "voltage_pred_v",
                "residual_pred_v",
                "voltage_hybrid_pred_v",
                "hybrid_error_v",
            ]
        ].head()
    )


    baseline_train_rmse = np.sqrt(
        mean_squared_error(train_table["voltage_v"], train_table["voltage_pred_v"])
    )
    baseline_test_rmse = np.sqrt(
        mean_squared_error(test_table["voltage_v"], test_table["voltage_pred_v"])
    )

    hybrid_train_rmse = np.sqrt(
        mean_squared_error(train_table["voltage_v"], train_table["voltage_hybrid_pred_v"])
    )
    hybrid_test_rmse = np.sqrt(
        mean_squared_error(test_table["voltage_v"], test_table["voltage_hybrid_pred_v"])
    )

    baseline_train_mae = mean_absolute_error(
        train_table["voltage_v"], train_table["voltage_pred_v"]
    )
    baseline_test_mae = mean_absolute_error(
        test_table["voltage_v"], test_table["voltage_pred_v"]
    )

    hybrid_train_mae = mean_absolute_error(
        train_table["voltage_v"], train_table["voltage_hybrid_pred_v"]
    )
    hybrid_test_mae = mean_absolute_error(
        test_table["voltage_v"], test_table["voltage_hybrid_pred_v"]
    )

    print("\nBaseline vs hybrid metrics:")
    print(f"Baseline train RMSE: {baseline_train_rmse:.6f} V")
    print(f"Hybrid train RMSE:   {hybrid_train_rmse:.6f} V")
    print(f"Baseline test RMSE:  {baseline_test_rmse:.6f} V")
    print(f"Hybrid test RMSE:    {hybrid_test_rmse:.6f} V")

    print(f"Baseline train MAE:  {baseline_train_mae:.6f} V")
    print(f"Hybrid train MAE:    {hybrid_train_mae:.6f} V")
    print(f"Baseline test MAE:   {baseline_test_mae:.6f} V")
    print(f"Hybrid test MAE:     {hybrid_test_mae:.6f} V")

    train_table["split"] = "train"
    test_table["split"] = "test"

    prediction_table = pd.concat([train_table, test_table], ignore_index=True)

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    prediction_path = output_dir / f"hybrid_residual_voltage_predictions_{battery_id}.csv"
    prediction_table.to_csv(prediction_path, index=False)

    metrics_table = pd.DataFrame(
        [
            {
                "model_name": "physics_inspired_baseline",
                "battery_id": battery_id,
                "train_rmse_v": baseline_train_rmse,
                "test_rmse_v": baseline_test_rmse,
                "train_mae_v": baseline_train_mae,
                "test_mae_v": baseline_test_mae,
            },
            {
                "model_name": "hybrid_residual_without_temperature",
                "battery_id": battery_id,
                "train_rmse_v": hybrid_train_rmse,
                "test_rmse_v": hybrid_test_rmse,
                "train_mae_v": hybrid_train_mae,
                "test_mae_v": hybrid_test_mae,
            },
        ]
    )

    metrics_path = output_dir / f"hybrid_residual_voltage_metrics_{battery_id}.csv"
    metrics_table.to_csv(metrics_path, index=False)

    print("\nSaved hybrid prediction CSV file:")
    print(prediction_path)

    print("\nSaved hybrid metrics CSV file:")
    print(metrics_path)

    plot_cycle = int(test_table["discharge_index"].iloc[0])
    plot_table = test_table[test_table["discharge_index"] == plot_cycle].copy()

    figure_dir = project_root / "figures"
    figure_dir.mkdir(exist_ok=True)

    figure_path = (
        figure_dir
        / f"hybrid_residual_voltage_prediction_{battery_id}_cycle_{plot_cycle}.png"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        plot_table["time_s"],
        plot_table["voltage_v"],
        marker="o",
        markersize=2,
        linewidth=1,
        label="Measured voltage",
    )

    plt.plot(
        plot_table["time_s"],
        plot_table["voltage_pred_v"],
        linewidth=2,
        label="Physics-inspired baseline",
    )

    plt.plot(
        plot_table["time_s"],
        plot_table["voltage_hybrid_pred_v"],
        linewidth=2,
        label="Hybrid residual prediction",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"{battery_id} hybrid residual voltage prediction, cycle {plot_cycle}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print("\nSaved hybrid figure:")
    print(figure_path)

    per_cycle_rows = []

    for discharge_index, cycle_table in test_table.groupby("discharge_index"):
        baseline_cycle_rmse = np.sqrt(
            mean_squared_error(
                cycle_table["voltage_v"],
                cycle_table["voltage_pred_v"],
            )
        )
        hybrid_cycle_rmse = np.sqrt(
            mean_squared_error(
                cycle_table["voltage_v"],
                cycle_table["voltage_hybrid_pred_v"],
            )
        )

        baseline_cycle_mae = mean_absolute_error(
            cycle_table["voltage_v"],
            cycle_table["voltage_pred_v"],
        )
        hybrid_cycle_mae = mean_absolute_error(
            cycle_table["voltage_v"],
            cycle_table["voltage_hybrid_pred_v"],
        )

        per_cycle_rows.append(
            {
                "battery_id": battery_id,
                "discharge_index": discharge_index,
                "number_of_steps": len(cycle_table),
                "baseline_rmse_v": baseline_cycle_rmse,
                "hybrid_rmse_v": hybrid_cycle_rmse,
                "baseline_mae_v": baseline_cycle_mae,
                "hybrid_mae_v": hybrid_cycle_mae,
                "rmse_improvement_v": baseline_cycle_rmse - hybrid_cycle_rmse,
                "mae_improvement_v": baseline_cycle_mae - hybrid_cycle_mae,
            }
        )

    per_cycle_metrics = pd.DataFrame(per_cycle_rows)

    per_cycle_metrics_path = (
        output_dir / f"hybrid_residual_voltage_per_cycle_metrics_{battery_id}.csv"
    )
    per_cycle_metrics.to_csv(per_cycle_metrics_path, index=False)

    print("\nSaved per-cycle metrics CSV file:")
    print(per_cycle_metrics_path)

    print("\nPer-cycle metrics preview:")
    print(per_cycle_metrics.head())

    print("\nWorst 5 hybrid test cycles by RMSE:")
    print(
        per_cycle_metrics.sort_values("hybrid_rmse_v", ascending=False).head()
    )

if __name__ == "__main__":
    main()

