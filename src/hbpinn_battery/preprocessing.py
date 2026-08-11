"""Reusable preprocessing for NASA battery discharge sequences."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_mat_file, matlab_array_to_1d, matlab_string_to_python


SEQUENCE_COLUMNS = [
    "battery_id",
    "discharge_index",
    "original_cycle_index",
    "step_index",
    "time_s",
    "dt_s",
    "current_a",
    "voltage_v",
    "temperature_c",
    "capacity_ah",
    "capacity_is_observed",
    "step_discharge_ah",
    "used_capacity_ah",
    "step_energy_kwh",
    "cycle_energy_kwh",
    "cumulative_energy_kwh",
]

SUMMARY_COLUMNS = [
    "battery_id",
    "mat_file",
    "number_of_discharge_cycles",
    "number_of_time_steps",
    "missing_capacity_cycles",
    "missing_capacity_time_steps",
    "final_cumulative_energy_kwh",
]


def trim_discharge_to_min_voltage(voltage_v, current_a, temperature_c, time_s):
    """Trim aligned discharge arrays through the first voltage minimum.

    ``voltage_v``, ``current_a``, ``temperature_c``, and ``time_s`` contain
    aligned voltage, current, temperature, and original time samples. The
    first minimum-voltage position is retained, after which all four arrays
    are aligned to the shortest remaining length. No current filtering or
    time re-zeroing is applied.

    Returns the arrays in voltage, current, temperature, and time order.
    """
    voltage_v = np.asarray(voltage_v)
    current_a = np.asarray(current_a)
    temperature_c = np.asarray(temperature_c)
    time_s = np.asarray(time_s)

    min_voltage_position = int(np.argmin(voltage_v))
    voltage_v = voltage_v[: min_voltage_position + 1]
    current_a = current_a[: min_voltage_position + 1]
    temperature_c = temperature_c[: min_voltage_position + 1]
    time_s = time_s[: min_voltage_position + 1]

    number_of_steps = min(
        len(time_s),
        len(voltage_v),
        len(current_a),
        len(temperature_c),
    )
    return (
        voltage_v[:number_of_steps],
        current_a[:number_of_steps],
        temperature_c[:number_of_steps],
        time_s[:number_of_steps],
    )


def extract_discharge_sequences(mat_files):
    """Extract discharge cycles from explicit NASA ``.mat`` file paths.

    The input is an iterable of paths; no fixed raw-data directory is used.
    Battery IDs are the file stems, and the implementation calculates time
    increments, discharged ampere-hours, and cycle and cumulative energy for
    each extracted sequence. Missing capacity is stored as ``NaN`` and
    identified by ``capacity_is_observed``.

    Returns ``(sequence_table, summary_table)``. The first table contains one
    row per discharge time step, while the second contains per-battery
    extraction counts and diagnostics. This function does not write CSV files
    or produce plots.
    """
    mat_file_by_name = {Path(path).stem: Path(path) for path in mat_files}
    battery_ids = sorted(mat_file_by_name.keys())
    all_rows = []
    summaries = []

    for battery_id in battery_ids:
        battery_file = mat_file_by_name[battery_id]
        mat_data = load_mat_file(battery_file)
        battery_object = mat_data[battery_id][0, 0]
        cycle_array = battery_object["cycle"].squeeze()

        rows = []
        discharge_index = 0
        cumulative_energy_kwh = 0.0
        missing_capacity_cycles = 0
        missing_capacity_time_steps = 0

        for original_cycle_index, cycle in enumerate(cycle_array):
            cycle_type = matlab_string_to_python(cycle["type"])
            if cycle_type != "discharge":
                continue

            data_object = cycle["data"][0, 0]
            voltage_v = matlab_array_to_1d(data_object["Voltage_measured"])
            current_a = matlab_array_to_1d(data_object["Current_measured"])
            temperature_c = matlab_array_to_1d(
                data_object["Temperature_measured"]
            )
            time_s = matlab_array_to_1d(data_object["Time"])

            voltage_v, current_a, temperature_c, time_s = (
                trim_discharge_to_min_voltage(
                    voltage_v,
                    current_a,
                    temperature_c,
                    time_s,
                )
            )

            capacity_array = matlab_array_to_1d(data_object["Capacity"])
            capacity_is_observed = len(capacity_array) > 0
            if capacity_is_observed:
                capacity_ah = float(capacity_array[0])
            else:
                capacity_ah = np.nan

            number_of_steps = len(time_s)
            if not capacity_is_observed:
                missing_capacity_cycles += 1
                missing_capacity_time_steps += number_of_steps

            dt_s = np.diff(time_s, prepend=time_s[0])
            dt_s = np.maximum(dt_s, 0.0)
            step_discharge_ah = np.abs(current_a) * dt_s / 3600.0
            used_capacity_ah = np.cumsum(step_discharge_ah)
            step_energy_kwh = (
                voltage_v * np.abs(current_a) * dt_s / 3_600_000.0
            )
            cycle_energy_kwh = float(np.sum(step_energy_kwh))
            cumulative_energy_within_cycle = (
                cumulative_energy_kwh + np.cumsum(step_energy_kwh)
            )

            for step_index in range(number_of_steps):
                rows.append(
                    {
                        "battery_id": battery_id,
                        "discharge_index": discharge_index,
                        "original_cycle_index": original_cycle_index,
                        "step_index": step_index,
                        "time_s": float(time_s[step_index]),
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
                        "cumulative_energy_kwh": float(
                            cumulative_energy_within_cycle[step_index]
                        ),
                    }
                )

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

    return (
        pd.DataFrame(all_rows, columns=SEQUENCE_COLUMNS),
        pd.DataFrame(summaries, columns=SUMMARY_COLUMNS),
    )
