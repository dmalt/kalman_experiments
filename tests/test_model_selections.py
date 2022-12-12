import numpy as np

from kalman_experiments import SSPE
from kalman_experiments.kalman.wrappers import PerturbedP1DMatsudaKF
from kalman_experiments.model_selection import fit_kf_parameters, normalize_measurement_dimensions
from kalman_experiments.models import (
    MatsudaParams,
    SingleRhythmModel,
    collect,
    gen_ar_noise_coefficients,
)
from kalman_experiments.numpy_types import Vec


def test_normalize_measurement_dimensions_prepends_nan():
    meas = np.ones(10)
    meas_norm = normalize_measurement_dimensions(meas)
    assert np.isnan(meas_norm[0][0, 0])


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


def test_model_fit_on_matsuda_data():
    mp = MatsudaParams(A=0.99, freq=10, sr=1000)
    gt_states = collect(SingleRhythmModel(mp, sigma=1), n_samp=4000)
    meas: Vec = np.real(gt_states) + 10 * np.random.randn(len(gt_states))  # type: ignore
    mp_init = MatsudaParams(A=0.99, freq=12, sr=1000)
    kf = PerturbedP1DMatsudaKF(mp_init, q_s=0.8, psi=np.zeros(1), r_s=5, lambda_=0)
    kf = fit_kf_parameters(meas, kf, tol=1e-3)
    assert abs(kf.mp.freq - 10) < 1, f"freq={kf.mp.freq}"


def test_model_fit_on_sspe_sines_w_pink():
    sim = SSPE.gen_sine_w_pink(1, 1000)
    a = gen_ar_noise_coefficients(alpha=1, order=20)
    mp_init = MatsudaParams(A=0.8, freq=1, sr=1000)
    kf = PerturbedP1DMatsudaKF(mp_init, q_s=1, psi=a, r_s=1, lambda_=1e-3)
    kf = fit_kf_parameters(sim.data, kf)
    assert abs(kf.mp.freq - 6) < 1
