from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    project_root = Path(__file__).resolve().parents[1]
    sequence_path = project_root / "results" / "discharge_sequences_all.csv"

    table = pd.read_csv(sequence_path)

    print("Loaded multi-battery discharge sequence dataset:")
    print(sequence_path)

    print("\nTable shape:")
    print(table.shape)

    print("\nNumber of batteries:")
    print(table["battery_id"].nunique())

    print("\nBattery ids:")
    print(sorted(table["battery_id"].unique()))

    print("\nRows per battery:")
    print(table["battery_id"].value_counts().sort_index())

    print("\nColumns:")
    print(table.columns.tolist())

    model_table = table[table["capacity_is_observed"]].copy()

    model_table = model_table.dropna(
        subset=[
            "capacity_ah",
            "used_capacity_ah",
            "current_a",
            "voltage_v",
            "temperature_c",
            "dt_s",
            "cumulative_energy_kwh",
        ]
    )

    print("\nModel table shape after filtering:")
    print(model_table.shape)

    print("\nNumber of batteries after filtering:")
    print(model_table["battery_id"].nunique())

    print("\nRows per battery after filtering:")
    print(model_table["battery_id"].value_counts().sort_index())


if __name__ == "__main__":
    main()