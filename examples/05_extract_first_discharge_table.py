from pathlib import Path

import pandas as pd

from hbpinn_battery.data import (
    find_mat_files,
    load_mat_file,
    matlab_array_to_1d,
    matlab_string_to_python,
)


def find_first_discharge_cycle(cycle_array):
    """
    Find the first discharge cycle inside a NASA battery cycle array.
    """
    for index, cycle in enumerate(cycle_array.flat):
        cycle_type = matlab_string_to_python(cycle["type"])

        if cycle_type == "discharge":
            return index, cycle

    raise ValueError("No discharge cycle was found.")


def main():
    """
    Extract the first discharge cycle from B0005 and save it as a CSV table.
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

    discharge_index, discharge_cycle = find_first_discharge_cycle(cycle_array)

    data_raw = discharge_cycle["data"]
    data_object = data_raw[0, 0]

    time_s = matlab_array_to_1d(data_object["Time"])
    voltage_measured_v = matlab_array_to_1d(data_object["Voltage_measured"])
    current_measured_a = matlab_array_to_1d(data_object["Current_measured"])
    temperature_measured_c = matlab_array_to_1d(data_object["Temperature_measured"])
    capacity_ah = float(matlab_array_to_1d(data_object["Capacity"])[0])

    table = pd.DataFrame(
        {
            "battery_id": battery_id,
            "cycle_index": discharge_index,
            "time_s": time_s,
            "voltage_measured_v": voltage_measured_v,
            "current_measured_a": current_measured_a,
            "temperature_measured_c": temperature_measured_c,
            "capacity_ah": capacity_ah,
        }
    )

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"first_discharge_{battery_id}.csv"
    table.to_csv(output_path, index=False)

    print(f"\nBattery id: {battery_id}")
    print(f"Discharge cycle index: {discharge_index}")
    print(f"Capacity: {capacity_ah:.6f} Ah")
    print(f"Number of time points: {len(table)}")

    print("\nFirst rows:")
    print(table.head())

    print("\nSaved CSV file:")
    print(output_path)


if __name__ == "__main__":
    main()