from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """
    Plot the capacity aging curve for B0005.
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
    final_capacity_ah = float(table["capacity_ah"].iloc[-1])
    eol_capacity_ah = 0.8 * initial_capacity_ah

    figure_dir = project_root / "figures"
    figure_dir.mkdir(exist_ok=True)

    figure_path = figure_dir / f"capacity_aging_{battery_id}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(
        table["discharge_index"],
        table["capacity_ah"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Measured capacity",
    )
    plt.axhline(
        eol_capacity_ah,
        linestyle="--",
        label="80% initial capacity",
    )
    plt.xlabel("Discharge cycle index")
    plt.ylabel("Capacity [Ah]")
    plt.title(
        f"{battery_id} capacity aging, initial {initial_capacity_ah:.3f} Ah, "
        f"final {final_capacity_ah:.3f} Ah"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print("Loaded CSV file:")
    print(csv_path)

    print(f"\nBattery id: {battery_id}")
    print(f"Number of discharge cycles: {len(table)}")
    print(f"Initial capacity: {initial_capacity_ah:.6f} Ah")
    print(f"Final capacity: {final_capacity_ah:.6f} Ah")
    print(f"80% EOL threshold: {eol_capacity_ah:.6f} Ah")

    print("\nSaved figure:")
    print(figure_path)


if __name__ == "__main__":
    main()