from pathlib import Path

from hbpinn_battery.data import find_mat_files, load_mat_file


def describe_value(name, value):
    """
    Print basic information about a Python object loaded from a MATLAB file.
    """
    print(f"\n{name}:")
    print(f"  type: {type(value)}")

    if hasattr(value, "shape"):
        print(f"  shape: {value.shape}")

    if hasattr(value, "dtype"):
        print(f"  dtype: {value.dtype}")

        if value.dtype.names is not None:
            print("  fields:")
            for field_name in value.dtype.names:
                print(f"    - {field_name}")


def main():
    """
    Load one NASA battery .mat file and inspect its nested structure.
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

    print("\nTop-level keys:")
    for key in mat_data.keys():
        print(f"  - {key}")

    public_keys = [key for key in mat_data.keys() if not key.startswith("__")]

    print("\nBattery data keys:")
    for key in public_keys:
        print(f"  - {key}")

    if not public_keys:
        print("No battery data key found.")
        return

    battery_key = public_keys[0]
    battery_object = mat_data[battery_key]

    print(f"\nSelected battery key: {battery_key}")
    describe_value("battery_object", battery_object)

    battery_scalar = battery_object[0, 0]
    describe_value("battery_scalar", battery_scalar)

    cycle_raw = battery_scalar["cycle"]
    describe_value("cycle_raw", cycle_raw)

    cycle_array = cycle_raw.squeeze()
    describe_value("cycle_array", cycle_array)

    number_of_cycles = cycle_array.size
    print(f"\nNumber of cycles: {number_of_cycles}")

    first_cycle = cycle_array.flat[0]
    describe_value("first_cycle", first_cycle)

    if hasattr(first_cycle, "dtype") and first_cycle.dtype.names is not None:
        print("\nFirst cycle fields:")
        for field_name in first_cycle.dtype.names:
            field_value = first_cycle[field_name]
            describe_value(f"first_cycle['{field_name}']", field_value)


if __name__ == "__main__":
    main()