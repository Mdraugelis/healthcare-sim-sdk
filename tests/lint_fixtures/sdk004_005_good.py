# lint-as: tests/unit/test_example.py
import numpy as np
import pytest


def test_assert():
    assert 1 + 1 == 2


def test_helper():
    np.testing.assert_array_equal([1], [1])


def test_raises():
    with pytest.raises(ZeroDivisionError, match="division"):
        1 / 0


def test_guard_with_precondition(rates={"Black": 0.1}):
    assert "Black" in rates
    if rates["Black"] > 0:
        assert rates["Black"] < 1
