from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def main():
    """
    Fit a simple quadratic baseline model for B0005 capacity aging.

    The first 70% of discharge cycles are used for training.
    The remaining 30% are used as a simple future-capacity test set.
    """
    project_root = Path(__file__).resolve().parents[1]

    csv_path = project_root / "results" / "capacity_aging_B0005.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Please run examples/07_extract_capacity_aging_table.py first."
        )

    table = pd.read_csv(csv_path)

    battery_id = table["battery_id"].iloc[0]
    initial_capacity_ah = float(table["capacity_ah"].iloc[0])
    eol_capacity_ah = 0.8 * initial_capacity_ah

    train_size = int(0.7 * len(table))
    train_table = table.iloc[:train_size].copy()
    test_table = table.iloc[train_size:].copy()

    X_train = train_table[["discharge_index"]]
    y_train = train_table["capacity_ah"]

    X_test = test_table[["discharge_index"]]
    y_test = test_table["capacity_ah"]

    X_all = table[["discharge_index"]]

    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )

    model.fit(X_train, y_train)

    table["capacity_pred_ah"] = model.predict(X_all)
    test_pred = model.predict(X_test)

    errors = test_pred - y_test.to_numpy()
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    prediction_path = output_dir / f"capacity_baseline_predictions_{battery_id}.csv"
    table.to_csv(prediction_path, index=False)

    figure_dir = project_root / "figures"
    figure_dir.mkdir(exist_ok=True)

    figure_path = figure_dir / f"capacity_baseline_{battery_id}.png"

    split_cycle = int(test_table["discharge_index"].iloc[0])

    plt.figure(figsize=(8, 5))
    plt.plot(
        table["discharge_index"],
        table["capacity_ah"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Measured capacity",
    )
    plt.plot(
        table["discharge_index"],
        table["capacity_pred_ah"],
        linewidth=2,
        label="Quadratic baseline prediction",
    )
    plt.axvline(
        split_cycle,
        linestyle=":",
        label="Train/test split",
    )
    plt.axhline(
        eol_capacity_ah,
        linestyle="--",
        label="80% initial capacity",
    )
    plt.xlabel("Discharge cycle index")
    plt.ylabel("Capacity [Ah]")
    plt.title(f"{battery_id} quadratic baseline, test RMSE {rmse:.4f} Ah")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print("Loaded CSV file:")
    print(csv_path)

    print(f"\nBattery id: {battery_id}")
    print(f"Number of discharge cycles: {len(table)}")
    print(f"Training points: {len(train_table)}")
    print(f"Test points: {len(test_table)}")
    print(f"Test RMSE: {rmse:.6f} Ah")
    print(f"Test MAE: {mae:.6f} Ah")

    print("\nSaved prediction CSV file:")
    print(prediction_path)

    print("\nSaved figure:")
    print(figure_path)


if __name__ == "__main__":
    main()