import numpy as np
import pytest
from scipy.stats import circstd

from kalman_experiments import SSPE
from kalman_experiments.kalman.wrappers import (
    Difference1DMatsudaKF,
    PerturbedP1DMatsudaKF,
    PerturbedP1DMatsudaSmoother,
    apply_kf,
)
from kalman_experiments.models import MatsudaParams, gen_ar_noise_coefficients


@pytest.mark.parametrize(
    "A, q_s, r_s, alpha, order, sim_func, bound",
    [
        (0.99999, 0.001, 0.2, 0, 1, SSPE.gen_sine_w_white, 1.8),
        (0.99999, 0.001, 0.2, 1.5, 30, SSPE.gen_sine_w_pink, 2.1),
        (0.99999, 0.001, 0.2, 0, 1, SSPE.gen_sine_w_pink, 10),
        (0.9995, 0.18, 0.14, 1.5, 30, SSPE.gen_filt_pink_noise_w_added_pink_noise, 60),
        (0.99, 3.16, 1, 0, 1, SSPE.gen_state_space_model_white, 45),
        (0.99, 3.16, 1, 1.5, 30, SSPE.gen_state_space_model_pink, 55),
        (0.99, 3.16, 1, 0, 1, SSPE.gen_state_space_model_pink, 55),
    ],
)
def test_sspe_metrics_good_enough_aug(A, q_s, r_s, alpha, order, sim_func, bound):
    srate = 250
    mp = MatsudaParams(A=A, freq=6, sr=srate)
    psi = gen_ar_noise_coefficients(alpha=alpha, order=order)
    sim = sim_func(duration_sec=10, Fs=srate)
    kf = PerturbedP1DMatsudaKF(mp, q_s=q_s, psi=psi, r_s=r_s, lambda_=0)

    filtered_data = apply_kf(kf, sim.data, delay=0)
    phase_diff = circstd(np.angle(filtered_data) - sim.true_phase) * 180 / np.pi
    assert phase_diff < bound


@pytest.mark.parametrize(
    "A, q_s, r_s, alpha, sim_func, bound",
    [
        (0.99999, 0.001, 0.2, 0, SSPE.gen_sine_w_white, 1.5),
        (0.99999, 0.001, 0.2, 1.5, SSPE.gen_sine_w_pink, 3),
        (0.9995, 0.18, 0.14, 1.5, SSPE.gen_filt_pink_noise_w_added_pink_noise, 60),
        (0.99, 3.16, 1, 0, SSPE.gen_state_space_model_white, 45),
        (0.99, 3.16, 1, 1.5, SSPE.gen_state_space_model_pink, 60),
    ],
)
def test_sspe_metrics_good_enough_diff(A, q_s, r_s, alpha, sim_func, bound):
    srate = 250
    mp = MatsudaParams(A=A, freq=6, sr=srate)
    psi = gen_ar_noise_coefficients(alpha=alpha, order=1)
    sim = sim_func(duration_sec=10, Fs=srate)
    kf = Difference1DMatsudaKF(mp, q_s=q_s, psi=psi[0], r_s=r_s)

    filtered_data = apply_kf(kf, sim.data, delay=0)
    phase_diff = circstd(np.angle(filtered_data) - sim.true_phase) * 180 / np.pi
    assert phase_diff < bound


@pytest.mark.parametrize(
    "A, q_s, r_s, alpha, order, sim_func",
    [
        (0.99999, 0.001, 0.2, 0, 1, SSPE.gen_sine_w_white),
        (0.99999, 0.001, 0.2, 1.5, 30, SSPE.gen_sine_w_pink),
        (0.99999, 0.001, 0.2, 0, 1, SSPE.gen_sine_w_pink),
        (0.9995, 0.18, 0.14, 1.5, 30, SSPE.gen_filt_pink_noise_w_added_pink_noise),
        (0.99, 3.16, 1, 0, 1, SSPE.gen_state_space_model_white),
        (0.99, 3.16, 1, 1.5, 30, SSPE.gen_state_space_model_pink),
        (0.99, 3.16, 1, 0, 1, SSPE.gen_state_space_model_pink),
    ],
)
def test_aug_smoother_better_than_kf(A, q_s, r_s, alpha, order, sim_func):
    sr = 250
    mp = MatsudaParams(A=A, freq=6, sr=sr)
    psi = gen_ar_noise_coefficients(alpha=alpha, order=order)
    sim = sim_func(duration_sec=10, Fs=sr)
    kf = PerturbedP1DMatsudaKF(mp, q_s=q_s, psi=psi, r_s=r_s, lambda_=0)
    delay = 10
    smoother = PerturbedP1DMatsudaSmoother(mp, q_s=q_s, psi=psi, r_s=r_s, lag=delay, lambda_=0)

    filtered_kf = apply_kf(kf, sim.data, delay=0)[:-delay]
    filtered_smoother = apply_kf(smoother, sim.data, delay=delay)[delay:]
    true_phase = sim.true_phase[:-delay]
    phase_diff_kf = circstd(np.angle(filtered_kf) - true_phase) * 180 / np.pi
    phase_diff_smoother = circstd(np.angle(filtered_smoother) - true_phase) * 180 / np.pi
    assert phase_diff_smoother < phase_diff_kf


@pytest.mark.parametrize(
    "A, q_s, r_s, sim_func",
    [
        (0.99999, 0.001, 0.2, SSPE.gen_sine_w_white),
        (0.99999, 0.001, 0.2, SSPE.gen_sine_w_pink),
        (0.99, 3.16, 1, SSPE.gen_state_space_model_white),
        (0.99, 3.16, 1, SSPE.gen_state_space_model_pink),
    ],
)
def test_diff_and_aug_kf_equivalent_for_zero_psi(A, q_s, r_s, sim_func):
    sr = 250
    mp = MatsudaParams(A=A, freq=6, sr=sr)
    sim = sim_func(duration_sec=10, Fs=sr)

    kf_aug = PerturbedP1DMatsudaKF(mp, q_s=q_s, psi=np.array([0]), r_s=r_s, lambda_=0)
    kf_diff = Difference1DMatsudaKF(mp, q_s=q_s, psi=0, r_s=r_s)

    filtered_aug = apply_kf(kf_aug, sim.data, delay=0)
    filtered_diff = apply_kf(kf_diff, sim.data, delay=0)
    np.testing.assert_allclose(filtered_aug, filtered_diff)


@pytest.mark.parametrize(
    "A, q_s, r_s, alpha, sim_func",
    [
        (0.99999, 0.001, 0.2, 0, SSPE.gen_sine_w_white),
        (0.99999, 0.001, 0.2, 1.5, SSPE.gen_sine_w_pink),
        (0.99999, 0.001, 0.2, 0, SSPE.gen_sine_w_pink),
        (0.9995, 0.18, 0.14, 1.5, SSPE.gen_filt_pink_noise_w_added_pink_noise),
        (0.99, 3.16, 1, 0, SSPE.gen_state_space_model_white),
        (0.99, 3.16, 1, 1.5, SSPE.gen_state_space_model_pink),
        (0.99, 3.16, 1, 0, SSPE.gen_state_space_model_pink),
    ],
)
def test_diff_and_aug_kf_close_for_first_order_noise_sim(A, q_s, r_s, alpha, sim_func):
    sr = 250
    mp = MatsudaParams(A=A, freq=6, sr=sr)
    psi = gen_ar_noise_coefficients(alpha=alpha, order=1)
    sim = sim_func(duration_sec=10, Fs=sr)

    kf_aug = PerturbedP1DMatsudaKF(mp, q_s=q_s, psi=psi, r_s=r_s, lambda_=0)
    kf_diff = Difference1DMatsudaKF(mp, q_s=q_s, psi=psi[0], r_s=r_s)

    skip = sr
    filtered_aug = apply_kf(kf_aug, sim.data, delay=0)[skip:]
    filtered_diff = apply_kf(kf_diff, sim.data, delay=0)[skip:]
    np.testing.assert_allclose(filtered_aug, filtered_diff, rtol=0.1, atol=0)
