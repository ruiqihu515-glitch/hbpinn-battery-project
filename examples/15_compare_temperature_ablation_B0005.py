from pathlib import Path

import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[1]
    battery_id = "B0005"

    no_temperature_path = (
        project_root
        / "results"
        / f"hybrid_residual_voltage_per_cycle_metrics_{battery_id}.csv"
    )

    temperature_path = (
        project_root
        / "results"
        / f"temperature_aware_hybrid_voltage_per_cycle_metrics_{battery_id}.csv"
    )

    no_temperature_table = pd.read_csv(no_temperature_path)
    temperature_table = pd.read_csv(temperature_path)

    print("Loaded no-temperature metrics:")
    print(no_temperature_path)

    print("\nLoaded temperature-aware metrics:")
    print(temperature_path)

    comparison_table = no_temperature_table[
        [
            "battery_id",
            "discharge_index",
            "number_of_steps",
            "hybrid_rmse_v",
            "hybrid_mae_v",
        ]
    ].rename(
        columns={
            "hybrid_rmse_v": "no_temperature_rmse_v",
            "hybrid_mae_v": "no_temperature_mae_v",
        }
    )

    temperature_selected = temperature_table[
        [
            "battery_id",
            "discharge_index",
            "hybrid_rmse_v",
            "hybrid_mae_v",
        ]
    ].rename(
        columns={
            "hybrid_rmse_v": "temperature_rmse_v",
            "hybrid_mae_v": "temperature_mae_v",
        }
    )

    comparison_table = comparison_table.merge(
        temperature_selected,
        on=["battery_id", "discharge_index"],
        how="inner",
    )

    comparison_table["temperature_gain_rmse_v"] = (
        comparison_table["no_temperature_rmse_v"]
        - comparison_table["temperature_rmse_v"]
    )

    comparison_table["temperature_gain_mae_v"] = (
        comparison_table["no_temperature_mae_v"]
        - comparison_table["temperature_mae_v"]
    )

    comparison_table["temperature_improved_rmse"] = (
        comparison_table["temperature_gain_rmse_v"] > 0
    )

    comparison_table["temperature_improved_mae"] = (
        comparison_table["temperature_gain_mae_v"] > 0
    )

    output_path = (
        project_root
        / "results"
        / f"temperature_ablation_comparison_{battery_id}.csv"
    )

    comparison_table.to_csv(output_path, index=False)

    print("\nTemperature ablation summary:")
    print("Number of test cycles:", len(comparison_table))

    print("\nMean no-temperature RMSE:")
    print(comparison_table["no_temperature_rmse_v"].mean())

    print("\nMean temperature-aware RMSE:")
    print(comparison_table["temperature_rmse_v"].mean())

    print("\nMean temperature RMSE gain:")
    print(comparison_table["temperature_gain_rmse_v"].mean())

    print("\nNumber of cycles improved by RMSE:")
    print(comparison_table["temperature_improved_rmse"].sum())

    print("\nMean no-temperature MAE:")
    print(comparison_table["no_temperature_mae_v"].mean())

    print("\nMean temperature-aware MAE:")
    print(comparison_table["temperature_mae_v"].mean())

    print("\nMean temperature MAE gain:")
    print(comparison_table["temperature_gain_mae_v"].mean())

    print("\nNumber of cycles improved by MAE:")
    print(comparison_table["temperature_improved_mae"].sum())

    print("\nWorst 5 temperature changes by RMSE gain:")
    print(
        comparison_table.sort_values(
            "temperature_gain_rmse_v",
            ascending=True,
        ).head()
    )

    print("\nBest 5 temperature changes by RMSE gain:")
    print(
        comparison_table.sort_values(
            "temperature_gain_rmse_v",
            ascending=False,
        ).head()
    )

    print("\nSaved comparison CSV file:")
    print(output_path)

if __name__ == "__main__":
    main()

# temperature did not provide additional predictive value beyond SOC, current, and aging-history features.
# For B0005, the hybrid residual model without temperature already reduced the test RMSE substantially. 
# Adding temperature did not further improve the result: 
#   the average per-cycle RMSE changed from 0.01943 V to 0.02042 V, and only 3 out of 51 test cycles improved. 
# This suggests that, for this single-battery setting, temperature is largely redundant with SOC, current, and aging-history features.