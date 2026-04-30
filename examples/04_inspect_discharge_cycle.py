from pathlib import Path

from hbpinn_battery.data import (
    find_mat_files,
    load_mat_file,
    matlab_string_to_python,
)


def describe_value(name, value):
    """
    Print basic information about an object loaded from a MATLAB file.
    """
    print(f"\n{name}:")
    print(f"  type: {type(value)}")

    if hasattr(value, "shape"):
        print(f"  shape: {value.shape}")

    if hasattr(value, "dtype"):
        print(f"  dtype: {value.dtype}")

    if hasattr(value, "squeeze"):
        squeezed = value.squeeze()
    else:
        squeezed = value

    if hasattr(squeezed, "flat") and hasattr(squeezed, "size"):
        preview_count = min(5, squeezed.size)
        preview = [squeezed.flat[i] for i in range(preview_count)]
        print(f"  first values: {preview}")
    else:
        print(f"  value: {squeezed}")


def main():
    """
    Find the first discharge cycle in B0005 and inspect its measured data.
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

    battery_key = public_keys[0]
    battery_object = mat_data[battery_key]
    battery_scalar = battery_object[0, 0]

    cycle_raw = battery_scalar["cycle"]
    cycle_array = cycle_raw.squeeze()

    discharge_cycle = None
    discharge_index = None

    for index, cycle in enumerate(cycle_array.flat):
        cycle_type = matlab_string_to_python(cycle["type"])

        if cycle_type == "discharge":
            discharge_cycle = cycle
            discharge_index = index
            break

    if discharge_cycle is None:
        print("No discharge cycle was found.")
        return

    print(f"\nBattery key: {battery_key}")
    print(f"Selected discharge cycle index: {discharge_index}")

    print("\nDischarge cycle fields:")
    for field_name in discharge_cycle.dtype.names:
        print(f"  - {field_name}")

    data_raw = discharge_cycle["data"]
    data_object = data_raw[0, 0]

    print("\nDischarge data fields:")
    for field_name in data_object.dtype.names:
        print(f"  - {field_name}")

    for field_name in data_object.dtype.names:
        field_value = data_object[field_name]
        describe_value(f"data['{field_name}']", field_value)


if __name__ == "__main__":
    main()