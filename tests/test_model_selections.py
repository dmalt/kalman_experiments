import numpy as np

from kalman_experiments.model_selection import normalize_measurement_dimensions


def test_normalize_measurement_dimensions_prepends_nan():
    meas = np.ones(10)
    meas_norm = normalize_measurement_dimensions(meas)
    assert np.isnan(meas_norm[0][0,0])


def test_normalize_measurement_dimensions_preserves_packs_values_in_arrays_of_right_shape():
    meas = np.random.rand(100)
    meas_norm = normalize_measurement_dimensions(meas)
    for m in meas_norm:
        assert m.ndim == 2
        assert m.shape == (1, 1)


def test_normalize_measurement_dimensions_preserves_values_with_Vec1D():
    meas = np.random.rand(100)
    meas_norm = normalize_measurement_dimensions(meas)
    for m, n in zip(meas, meas_norm[1:]):
        assert m == n[0, 0]


def test_normalize_measurement_dimensions_preserves_values_with_Vec():
    meas = np.random.rand(100, 1)
    meas_norm = normalize_measurement_dimensions(meas)
    for m, n in zip(meas, meas_norm[1:]):
        assert m == n[0, 0]
