from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """
    Plot the voltage curve of the first extracted discharge cycle.

    The NASA discharge cycle can include a voltage recovery tail after the load
    is removed. For a clean discharge plot, we keep the data only until the
    minimum measured voltage.
    """
    project_root = Path(__file__).resolve().parents[1]

    csv_path = project_root / "results" / "first_discharge_B0005.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Please run examples/05_extract_first_discharge_table.py first."
        )

    table = pd.read_csv(csv_path)

    battery_id = table["battery_id"].iloc[0]
    cycle_index = int(table["cycle_index"].iloc[0])
    capacity_ah = float(table["capacity_ah"].iloc[0])

    min_voltage_index = table["voltage_measured_v"].idxmin()
    trimmed_table = table.loc[:min_voltage_index].copy()

    figure_dir = project_root / "figures"
    figure_dir.mkdir(exist_ok=True)

    figure_path = figure_dir / f"first_discharge_voltage_trimmed_{battery_id}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(
        trimmed_table["time_s"],
        trimmed_table["voltage_measured_v"],
        label="Measured voltage, trimmed",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(
        f"{battery_id}, discharge cycle {cycle_index}, capacity {capacity_ah:.3f} Ah"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print("Loaded CSV file:")
    print(csv_path)

    print("\nOriginal number of points:")
    print(len(table))

    print("\nTrimmed number of points:")
    print(len(trimmed_table))

    print("\nMinimum voltage:")
    print(trimmed_table["voltage_measured_v"].min())

    print("\nSaved figure:")
    print(figure_path)


if __name__ == "__main__":
    main()