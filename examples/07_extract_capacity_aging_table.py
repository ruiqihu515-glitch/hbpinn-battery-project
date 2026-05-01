from pathlib import Path

import pandas as pd

from hbpinn_battery.data import (
    find_mat_files,
    load_mat_file,
    matlab_array_to_1d,
    matlab_string_to_python,
)


def main():
    """
    Extract all discharge capacities from B0005 and save an aging table.
    """
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    data_dir = (
        workspace_root
        / "nasa_raw_data"
        / "5. Battery Data Set"
        / "extracted"
    )

    mat_files = find_mat_files(data_dir)

    if not mat_files:
        print("No .mat files were found.")
        return

    first_file = mat_files[0]

    print("Reading file:")
    print(first_file)

    mat_data = load_mat_file(first_file)

    public_keys = [key for key in mat_data.keys() if not key.startswith("__")]

    if not public_keys:
        print("No battery key was found in the .mat file.")
        return

    battery_id = public_keys[0]
    battery_object = mat_data[battery_id]
    battery_scalar = battery_object[0, 0]

    cycle_raw = battery_scalar["cycle"]
    cycle_array = cycle_raw.squeeze()

    rows = []
    discharge_index = 0

    for original_cycle_index, cycle in enumerate(cycle_array.flat):
        cycle_type = matlab_string_to_python(cycle["type"])

        if cycle_type != "discharge":
            continue

        data_raw = cycle["data"]
        data_object = data_raw[0, 0]

        capacity_ah = float(matlab_array_to_1d(data_object["Capacity"])[0])

        rows.append(
            {
                "battery_id": battery_id,
                "discharge_index": discharge_index,
                "original_cycle_index": original_cycle_index,
                "capacity_ah": capacity_ah,
            }
        )

        discharge_index += 1

    table = pd.DataFrame(rows)

    if table.empty:
        print("No discharge capacity data was found.")
        return

    initial_capacity_ah = table["capacity_ah"].iloc[0]
    table["relative_capacity"] = table["capacity_ah"] / initial_capacity_ah

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"capacity_aging_{battery_id}.csv"
    table.to_csv(output_path, index=False)

    print(f"\nBattery id: {battery_id}")
    print(f"Number of discharge cycles: {len(table)}")
    print(f"Initial capacity: {initial_capacity_ah:.6f} Ah")
    print(f"Final capacity: {table['capacity_ah'].iloc[-1]:.6f} Ah")

    print("\nFirst rows:")
    print(table.head())

    print("\nLast rows:")
    print(table.tail())

    print("\nSaved CSV file:")
    print(output_path)


if __name__ == "__main__":
    main()