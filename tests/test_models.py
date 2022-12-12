import numpy as np
from scipy.signal import welch

from kalman_experiments.models import ArNoiseModel, MatsudaParams, SingleRhythmModel, collect


def test_theoretical_psd_for_single_rhythm_model_matches_estimated():
    mp = MatsudaParams(A=0.99, freq=10, sr=1000)
    srm = SingleRhythmModel(mp, sigma=1)

    sim_data = collect(srm, n_samp=100_000)
    ff, psd_est = welch(np.real(sim_data), fs=mp.sr, nperseg=3500)
    ff_trim = ff[3:-1]
    psd_est_trim = psd_est[3:-1]
    theor_psd = np.array([srm.psd_onesided(f) for f in ff_trim])

    relative_error = psd_est_trim / theor_psd - 1
    assert np.std(relative_error) < 0.5
    assert np.abs(np.mean(relative_error)) < 0.05


def test_theoretical_psd_for_ar_noise_model_matches_estimated():
    order = 30
    sr = 1000
    srm = ArNoiseModel(x0=np.random.randn(order), order=order, alpha=1.5, sigma=1, sr=sr)

    sim_data = collect(srm, n_samp=100_000)
    ff, psd_est = welch(np.real(sim_data), fs=sr, nperseg=3500)
    ff_trim = ff[3:-1]
    psd_est_trim = psd_est[3:-1]
    theor_psd = np.array([srm.psd_onesided(f) for f in ff_trim])

    relative_error = psd_est_trim / theor_psd - 1
    assert np.std(relative_error) < 0.5
    assert np.abs(np.mean(relative_error)) < 0.05
