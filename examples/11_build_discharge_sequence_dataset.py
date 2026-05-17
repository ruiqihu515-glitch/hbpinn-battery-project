from pathlib import Path
import numpy as np
import pandas as pd
from hbpinn_battery.data import (
    find_mat_files,
    load_mat_file,
    matlab_array_to_1d,
    matlab_string_to_python,
)

def main():
    
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    data_dir = (
        workspace_root
        / "nasa_raw_data"
        / "5. Battery Data Set"
        / "extracted"
    )

    # find B0005.mat
    mat_files = find_mat_files(data_dir)
    mat_file_by_name = {path.stem: path for path in mat_files}

    battery_ids = sorted(mat_file_by_name.keys())

    # collect rows from all selected batteries
    all_rows = []
    summaries = []

    for battery_id in battery_ids:
        if battery_id not in mat_file_by_name:
            raise FileNotFoundError(f"Could not find {battery_id}.mat in {data_dir}")

        battery_file = mat_file_by_name[battery_id]

        # read one MATLAB battery file
        mat_data = load_mat_file(battery_file)
        battery_object = mat_data[battery_id][0, 0]
        cycle_array = battery_object["cycle"].squeeze()

        rows = []
        discharge_index = 0
        cumulative_energy_kwh = 0.0
        missing_capacity_cycles = 0
        missing_capacity_time_steps = 0

        # deal with all cycles
        for original_cycle_index, cycle in enumerate(cycle_array):
            cycle_type = matlab_string_to_python(cycle["type"])
            if cycle_type != "discharge":
                continue

            # recive disacharge data
            data_object = cycle["data"][0,0]
            voltage_v = matlab_array_to_1d(data_object["Voltage_measured"])
            current_a = matlab_array_to_1d(data_object["Current_measured"])
            temperature_c = matlab_array_to_1d(data_object["Temperature_measured"])
            time_s = matlab_array_to_1d(data_object["Time"])

            capacity_array = matlab_array_to_1d(data_object["Capacity"])

            capacity_is_observed = len(capacity_array) > 0

            if capacity_is_observed:
                capacity_ah = float(capacity_array[0])
            else:
                capacity_ah = np.nan

            number_of_steps = min(
                len(time_s),
                len(voltage_v),
                len(current_a),
                len(temperature_c),
            )

            time_s = time_s[:number_of_steps]
            voltage_v = voltage_v[:number_of_steps]
            current_a = current_a[:number_of_steps]
            temperature_c = temperature_c[:number_of_steps]
            if not capacity_is_observed:
                missing_capacity_cycles += 1
                missing_capacity_time_steps += number_of_steps

            dt_s = np.diff(time_s, prepend=time_s[0])
            dt_s = np.maximum(dt_s, 0.0)

            step_discharge_ah = np.abs(current_a) * dt_s / 3600.0
            used_capacity_ah = np.cumsum(step_discharge_ah)
            step_energy_kwh = voltage_v * np.abs(current_a) * dt_s / 3_600_000.0
            cycle_energy_kwh = float(np.sum(step_energy_kwh))
            cumulative_energy_within_cycle = cumulative_energy_kwh + np.cumsum(step_energy_kwh)

            # save each time step to one line
            for step_index in range(number_of_steps):
                rows.append(
                    {
                        "battery_id": battery_id,
                        "discharge_index": discharge_index,
                        "original_cycle_index": original_cycle_index,
                        "step_index": step_index,
                        "time_s":float(time_s[step_index]),
                        "dt_s": float(dt_s[step_index]),
                        "current_a": float(current_a[step_index]),
                        "voltage_v": float(voltage_v[step_index]),
                        "temperature_c": float(temperature_c[step_index]),
                        "capacity_ah": capacity_ah,
                        "capacity_is_observed": capacity_is_observed,
                        "step_discharge_ah": float(step_discharge_ah[step_index]),
                        "used_capacity_ah": float(used_capacity_ah[step_index]),
                        "step_energy_kwh": float(step_energy_kwh[step_index]),
                        "cycle_energy_kwh": cycle_energy_kwh,
                        "cumulative_energy_kwh": float(cumulative_energy_within_cycle[step_index])

                    }
                )
            
            # upadate cumulative energy
            cumulative_energy_kwh += cycle_energy_kwh
            discharge_index += 1

        all_rows.extend(rows)

        summaries.append(
            {
                "battery_id": battery_id,
                "mat_file": str(battery_file),
                "number_of_discharge_cycles": discharge_index,
                "number_of_time_steps": len(rows),
                "missing_capacity_cycles": missing_capacity_cycles,
                "missing_capacity_time_steps": missing_capacity_time_steps,
                "final_cumulative_energy_kwh": cumulative_energy_kwh,
            }
        )

    # save as .csv
    table = pd.DataFrame(all_rows)
    summary_table = pd.DataFrame(summaries)
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "discharge_sequences_all.csv"
    summary_path = output_dir / "discharge_sequences_all_summary.csv"
    table.to_csv(output_path, index=False)
    summary_table.to_csv(summary_path, index=False)

    # check
    print("Batteries found:")
    print(battery_ids)

    print("\nSummary:")
    print(summary_table)

    print("\nRows per battery:")
    print(table["battery_id"].value_counts())

    print("\nFirst rows:")
    print(table.head())

    print("\nLast rows:")
    print(table.tail())

    print("\nSaved sequence CSV file:")
    print(output_path)

    print("\nSaved summary CSV file:")
    print(summary_path)


if __name__ == "__main__":
    main()










    