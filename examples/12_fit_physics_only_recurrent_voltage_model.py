from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    project_root = Path(__file__).resolve().parents[1]
    sequence_path = project_root / "results" / "discharge_sequences_all.csv"
    table = pd.read_csv(sequence_path)
    print("loaded sequence file:")
    print(sequence_path)
    print("\nTable shape:")
    print(table.shape)

    # B0005
    battery_id = "B0005"
    battery_table = table[table["battery_id"] == battery_id].copy()
    print(f"\nBattery id:")
    print(battery_id)
    print("\nBattery table shape:")
    print(battery_table.shape)
    print("\nNumber of discharge cycles:")
    print(battery_table["discharge_index"].nunique())
    print("\nFirst rows of selected battery:")
    print(battery_table.head())

    model_table = battery_table[
        battery_table["capacity_is_observed"]
    ].copy()

    model_table = model_table.dropna(
        subset=[
            "capacity_ah",
            "used_capacity_ah",
            "current_a",
            "voltage_v",
            "dt_s",
        ]
    )

    print("\nModel table shape after filtering:")
    print(model_table.shape)    

    model_table["soc"] = 1.0 - model_table["used_capacity_ah"] / model_table["capacity_ah"]
    model_table["soc"] = model_table["soc"].clip(lower=0.0, upper=1.0)
    model_table["discharge_current_a"] = (-model_table["current_a"]).clip(lower=0.0)
    print("\nSOC range:")
    print(model_table["soc"].min(), model_table["soc"].max())
    print("\nDischarge current range:")
    print(model_table["discharge_current_a"].min(), model_table["discharge_current_a"].max())

    model_table["soc_squared"] = model_table["soc"] ** 2
    model_table["soc_cubed"] = model_table["soc"] ** 3

    feature_columns = [
        "soc",
        "soc_squared",
        "soc_cubed",
        "discharge_current_a",
    ]

    X = model_table[feature_columns]
    y = model_table["voltage_v"]

    print("\nFeature columns:")
    print(feature_columns)

    print("\nX shape:")
    print(X.shape)

    print("\ny shape:")
    print(y.shape)

    all_cycles = sorted(model_table["discharge_index"].unique())
    split_index = int(0.7 * len(all_cycles))

    train_cycles = all_cycles[:split_index]
    test_cycles = all_cycles[split_index:]

    train_table = model_table[model_table["discharge_index"].isin(train_cycles)].copy()
    test_table = model_table[model_table["discharge_index"].isin(test_cycles)].copy()

    X_train = train_table[feature_columns]
    y_train = train_table["voltage_v"]

    X_test = test_table[feature_columns]
    y_test = test_table["voltage_v"]

    print("\nNumber of all cycles:")
    print(len(all_cycles))

    print("\nNumber of train cycles:")
    print(len(train_cycles))

    print("\nNumber of test cycles:")
    print(len(test_cycles))

    print("\nTrain shape:")
    print(X_train.shape, y_train.shape)

    print("\nTest shape:")
    print(X_test.shape, y_test.shape)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test,y_test_pred)

    print("\nModel intercept:")
    print(model.intercept_)

    print("\nModel coefficients:")
    for feature_name, coefficient in zip(feature_columns, model.coef_):
        print(f"{feature_name}: {coefficient:.6f}")

    print("\nTrain RMSE:")
    print(f"{train_rmse:.6f} V")

    print("\nTest RMSE:")
    print(f"{test_rmse:.6f} V")

    print("\nTrain MAE:")
    print(f"{train_mae:.6f} V")

    print("\nTest MAE:")
    print(f"{test_mae:.6f} V")

    train_table["voltage_pred_v"] = y_train_pred
    test_table["voltage_pred_v"] = y_test_pred

    train_table["voltage_error_v"] = train_table["voltage_pred_v"] - train_table["voltage_v"]
    test_table["voltage_error_v"] = test_table["voltage_pred_v"] - test_table["voltage_v"]

    train_table["absolute_error_v"] = train_table["voltage_error_v"].abs()
    test_table["absolute_error_v"] = test_table["voltage_error_v"].abs()

    print("\nTest prediction preview:")
    print(
        test_table[
            [
                "battery_id",
                "discharge_index",
                "step_index",
                "soc",
                "discharge_current_a",
                "voltage_v",
                "voltage_pred_v",
                "voltage_error_v",
                "absolute_error_v",
            ]
        ].head()
    )


    train_table["split"] = "train"
    test_table["split"] = "test"

    prediction_table = pd.concat([train_table, test_table], ignore_index=True)

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    prediction_path = output_dir / f"physics_only_voltage_predictions_{battery_id}.csv"
    prediction_table.to_csv(prediction_path, index=False)

    print("\nSaved prediction CSV file:")
    print(prediction_path)

    plot_cycle = test_cycles[0]
    plot_table = test_table[test_table["discharge_index"] == plot_cycle].copy()

    figure_dir = project_root / "figures"
    figure_dir.mkdir(exist_ok=True)

    figure_path = figure_dir / f"physics_only_voltage_prediction_{battery_id}_cycle_{plot_cycle}.png"

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
        label="Physics-only prediction",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"{battery_id} physics-only voltage prediction, cycle {plot_cycle}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print("\nSaved figure:")
    print(figure_path)

    metrics_table = pd.DataFrame(
        [
            {
                "model_name": "physics_only_recurrent_voltage",
                "battery_id": battery_id,
                "number_of_train_cycles": len(train_cycles),
                "number_of_test_cycles": len(test_cycles),
                "number_of_train_rows": len(train_table),
                "number_of_test_rows": len(test_table),
                "train_rmse_v": train_rmse,
                "test_rmse_v": test_rmse,
                "train_mae_v": train_mae,
                "test_mae_v": test_mae,
                "intercept": model.intercept_,
                "coef_soc": model.coef_[0],
                "coef_soc_squared": model.coef_[1],
                "coef_soc_cubed": model.coef_[2],
                "coef_discharge_current_a": model.coef_[3],
            }
        ]
    )

    metrics_path = output_dir / f"physics_only_voltage_metrics_{battery_id}.csv"
    metrics_table.to_csv(metrics_path, index=False)

    print("\nSaved metrics CSV file:")
    print(metrics_path)



if __name__ == "__main__":
    main()
    
