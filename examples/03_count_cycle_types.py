from collections import Counter
from pathlib import Path
from hbpinn_battery.data import find_mat_files, load_mat_file

#  convert matlab string to python
from collections import Counter
from pathlib import Path

from hbpinn_battery.data import (
    find_mat_files,
    load_mat_file,
    matlab_string_to_python,
)

def main():
    """
    Count charge, discharge, and impedance cycles in one NASA battery file.
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
    cycle_type_counter = Counter()

    for cycle in cycle_array.flat:
        cycle_type_raw = cycle["type"]
        cycle_type = matlab_string_to_python(cycle_type_raw)
        cycle_type_counter[cycle_type] += 1
    
    print(f"\nBattery key: {battery_key}")
    print(f"Number of cycles: {cycle_array.size}")

    print("\nCycle type counts:")
    for cycle_type, count in cycle_type_counter.items():
           print(f"   - {cycle_type}: {count}")
    
if __name__ == "__main__":
       main()
           