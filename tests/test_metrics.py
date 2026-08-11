import numpy as np
import pytest

from hbpinn_battery.metrics import mae, rmse


def test_identical_arrays_have_zero_rmse():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_identical_arrays_have_zero_mae():
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_known_case():
    assert rmse([1.0, 2.0], [1.0, 3.0]) == pytest.approx(np.sqrt(0.5))


def test_mae_known_case():
    assert mae([1.0, 2.0], [1.0, 3.0]) == 0.5


def test_python_lists_are_accepted():
    assert rmse([0, 2], [0, 0]) == pytest.approx(np.sqrt(2.0))
    assert mae([0, 2], [0, 0]) == 1.0


def test_numpy_arrays_are_accepted():
    true = np.array([1.0, 4.0])
    pred = np.array([2.0, 2.0])
    assert rmse(true, pred) == pytest.approx(np.sqrt(2.5))
    assert mae(true, pred) == 1.5


@pytest.mark.parametrize("metric", [rmse, mae])
def test_mismatched_shapes_raise_value_error(metric):
    with pytest.raises(ValueError):
        metric([1.0, 2.0], [[1.0, 2.0]])


@pytest.mark.parametrize("metric", [rmse, mae])
def test_return_type_is_python_float(metric):
    assert type(metric([1.0, 2.0], [1.0, 3.0])) is float
