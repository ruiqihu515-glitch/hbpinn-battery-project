from pathlib import Path

import numpy as np
import pytest

import hbpinn_battery.preprocessing as preprocessing
from hbpinn_battery.preprocessing import (
    SEQUENCE_COLUMNS,
    extract_discharge_sequences,
    trim_discharge_to_min_voltage,
)


class FakeRecord:
    def __init__(self, **fields):
        self.fields = fields

    def __getitem__(self, key):
        return self.fields[key]


def fake_cycle(cycle_type, voltage, current, temperature, time, capacity):
    data = FakeRecord(
        Voltage_measured=np.asarray(voltage),
        Current_measured=np.asarray(current),
        Temperature_measured=np.asarray(temperature),
        Time=np.asarray(time),
        Capacity=np.asarray(capacity),
    )
    return FakeRecord(
        type=cycle_type,
        data=np.array([[data]], dtype=object),
    )


def fake_mat_data(battery_id):
    cycles = [
        fake_cycle("charge", [4.1], [1.0], [20.0], [0.0], []),
        fake_cycle(
            "discharge",
            [4.0, 3.0, 2.0, 2.5],
            [-2.0, -2.0, -2.0, 0.0],
            [20.0, 21.0, 22.0, 23.0],
            [10.0, 12.0, 15.0, 18.0],
            [1.8],
        ),
        fake_cycle(
            "discharge",
            [4.2, 3.8, 3.0, 2.5],
            [-1.0, -1.5],
            [30.0, 31.0, 32.0],
            [20.0, 21.0, 22.0, 23.0],
            [],
        ),
    ]
    battery = FakeRecord(cycle=np.asarray(cycles, dtype=object))
    return {battery_id: np.array([[battery]], dtype=object)}


def test_trim_through_minimum_voltage_without_rezeroing_time():
    voltage, current, temperature, time = trim_discharge_to_min_voltage(
        [4.0, 3.0, 2.0, 2.5],
        [-2.0, -2.0, -2.0, 0.0],
        [20.0, 21.0, 22.0, 23.0],
        [10.0, 12.0, 15.0, 18.0],
    )
    assert voltage.tolist() == [4.0, 3.0, 2.0]
    assert current.tolist() == [-2.0, -2.0, -2.0]
    assert temperature.tolist() == [20.0, 21.0, 22.0]
    assert time.tolist() == [10.0, 12.0, 15.0]


def test_trim_aligns_incomplete_arrays_and_empty_voltage_errors():
    arrays = trim_discharge_to_min_voltage(
        [4.0, 3.0, 2.0], [-1.0, -1.0], [20.0, 21.0, 22.0], [0.0, 1.0, 2.0]
    )
    assert all(len(array) == 2 for array in arrays)
    with pytest.raises(ValueError):
        trim_discharge_to_min_voltage([], [], [], [])


@pytest.fixture
def extracted(monkeypatch, tmp_path):
    battery_id = "BTEST"
    mat_path = tmp_path / f"{battery_id}.mat"
    monkeypatch.setattr(
        preprocessing,
        "load_mat_file",
        lambda path: fake_mat_data(Path(path).stem),
    )
    return extract_discharge_sequences([mat_path])


def test_extract_preserves_schema_order_cycles_capacity_and_temperature(extracted):
    table, summary = extracted
    assert table.columns.tolist() == SEQUENCE_COLUMNS
    assert table["discharge_index"].tolist() == [0, 0, 0, 1, 1]
    assert table["original_cycle_index"].tolist() == [1, 1, 1, 2, 2]
    assert table["step_index"].tolist() == [0, 1, 2, 0, 1]
    assert table.loc[:2, "capacity_ah"].tolist() == [1.8, 1.8, 1.8]
    assert table.loc[:2, "temperature_c"].tolist() == [20.0, 21.0, 22.0]
    assert table.loc[3:, "temperature_c"].tolist() == [30.0, 31.0]
    assert table.loc[3:, "capacity_ah"].isna().all()
    assert summary.loc[0, "number_of_discharge_cycles"] == 2
    assert summary.loc[0, "missing_capacity_cycles"] == 1
    assert summary.loc[0, "missing_capacity_time_steps"] == 2


def test_extract_preserves_time_dt_charge_and_energy_formulas(extracted):
    table, summary = extracted
    first = table.iloc[:3]
    assert first["time_s"].tolist() == [10.0, 12.0, 15.0]
    assert first["dt_s"].tolist() == [0.0, 2.0, 3.0]
    expected_step_charge = np.array([0.0, 4.0, 6.0]) / 3600.0
    assert np.allclose(first["step_discharge_ah"], expected_step_charge)
    assert np.allclose(first["used_capacity_ah"], np.cumsum(expected_step_charge))
    expected_step_energy = np.array([0.0, 3.0 * 2.0 * 2.0, 2.0 * 2.0 * 3.0]) / 3_600_000.0
    assert np.allclose(first["step_energy_kwh"], expected_step_energy)
    assert np.allclose(first["cycle_energy_kwh"], expected_step_energy.sum())
    assert np.allclose(first["cumulative_energy_kwh"], np.cumsum(expected_step_energy))
    second = table.iloc[3:]
    assert second["dt_s"].tolist() == [0.0, 1.0]
    assert second["cumulative_energy_kwh"].iloc[0] == pytest.approx(
        expected_step_energy.sum()
    )
    assert summary.loc[0, "final_cumulative_energy_kwh"] == pytest.approx(
        table.groupby("discharge_index")["cycle_energy_kwh"].first().sum()
    )


def test_extract_orders_multiple_batteries_by_file_stem(monkeypatch, tmp_path):
    paths = [tmp_path / "B0002.mat", tmp_path / "B0001.mat"]
    monkeypatch.setattr(
        preprocessing,
        "load_mat_file",
        lambda path: fake_mat_data(Path(path).stem),
    )
    table, summary = extract_discharge_sequences(paths)
    assert summary["battery_id"].tolist() == ["B0001", "B0002"]
    assert table["battery_id"].drop_duplicates().tolist() == ["B0001", "B0002"]


def test_import_and_execution_do_not_require_raw_data():
    table, summary = extract_discharge_sequences([])
    assert table.empty
    assert summary.empty
    assert table.columns.tolist() == SEQUENCE_COLUMNS
