"""
Simulations from SSPE paper

Examples
--------
Generate 10 seconds of data sampled at 1000 Hz and plot the PSD
>>> import matplotlib.pyplot as plt
>>> from scipy.signal import welch
>>> duration, fs, freq_lim = 10, 1000, 51
>>> data_wn, gt_wn, true_phase_wn = gen_sine_w_white(duration, fs)
>>> data_pn, gt_pn, true_phase_pn = gen_sine_w_pink(duration, fs)
>>> # broadband oscillation in pink noise
>>> data_pn_bb, gt_pn_bb, true_phase_pn_bb = gen_filt_pink_noise_w_added_pink_noise(duration, fs)
>>> freqs, psd_wn = welch(data_wn, fs=1000, nperseg=1000)
>>> _, psd_pn = welch(data_pn, fs=1000, nperseg=1000)
>>> _, psd_pn_bb = welch(data_pn_bb, fs=1000, nperseg=1000)
>>> ax = plt.plot(freqs[:freq_lim], 10 * np.log10(psd_wn)[:freq_lim], label="sine in white noise")
>>> ax = plt.plot(freqs[:freq_lim], 10 * np.log10(psd_pn)[:freq_lim], label="sine in pink noise")
>>> ax = plt.plot(freqs[:freq_lim], 10 * np.log10(psd_pn_bb)[:freq_lim], label="filt pink noise")
>>> legend = plt.legend()
>>> plt.grid()
>>> plt.show()

"""
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt, firwin, hilbert

from ..models import ArNoise, MatsudaParams, SingleRhythmModel, collect


def make_pink_noise(alpha: float, n_samp: int, dt: float) -> npt.NDArray[np.floating]:
    """Given an alpha value for the 1/f^alpha produce data of length n_samp and at Fs = 1/dt"""
    x1 = np.random.randn(n_samp)
    xf1 = np.fft.fft(x1)
    A = np.abs(xf1)
    phase = np.angle(xf1)

    df = 1 / (dt * len(x1))
    faxis = np.arange(len(x1) // 2 + 1) * df
    faxis = np.concatenate([faxis, faxis[-2:0:-1]])
    faxis[0] = np.inf

    oneOverf = 1 / faxis**alpha

    Anew = np.sqrt((A**2) * oneOverf.T)
    xf1new = Anew * np.exp(1j * phase)
    return np.real(np.fft.ifft(xf1new)).T


class SimulationResults(NamedTuple):
    data: npt.NDArray[np.floating]
    gt_states: npt.NDArray[np.complex_]
    true_phase: npt.NDArray[np.floating]


def gen_sine_w_white(duration_sec: float, Fs: float) -> SimulationResults:
    """Generate sine in white noise"""
    FREQ_HZ = 6
    A = 10

    n_samp = int(duration_sec * Fs)
    times = np.arange(1, n_samp + 1) / Fs
    true_phase = 2 * np.pi * FREQ_HZ * times
    Vlo = A * np.cos(true_phase)
    data = Vlo + np.random.randn(n_samp)
    true_phase = _wrapToPi(true_phase)
    gt_states = Vlo + 1j * A * np.sin(true_phase)
    return SimulationResults(data, gt_states, true_phase)


def gen_sine_w_pink(duration_sec: float, Fs: float) -> SimulationResults:
    """Generate sine in pink noise"""
    FREQ_HZ = 6
    A = 10
    PINK_NOISE_SNR = 10
    PINK_NOISE_ALPHA = 1.5

    n_samp = int(duration_sec * Fs)
    noise = make_pink_noise(PINK_NOISE_ALPHA, n_samp, 1 / Fs)
    times = np.arange(1, n_samp + 1) / Fs
    true_phase = 2 * np.pi * FREQ_HZ * times
    Vlo = A * np.cos(true_phase)
    data = Vlo + PINK_NOISE_SNR * noise
    true_phase = _wrapToPi(true_phase)
    gt_states = Vlo + 1j * A * np.sin(true_phase)
    return SimulationResults(data, gt_states, true_phase)


def gen_filt_pink_noise_w_added_pink_noise(duration_sec: float, Fs: float) -> SimulationResults:
    """
    Generate broadband oscillation in pink noise

    To get the broadband oscillation, filtfilt pink noise with FIR bandpass filter

    """
    A = 10
    FIR_ORDER = 750
    FIR_BAND_HZ = [4, 8]
    PINK_NOISE_SNR = 10
    PINK_NOISE_ALPHA = 1.5

    n_samp = int(duration_sec * Fs)
    b = firwin(numtaps=FIR_ORDER + 1, cutoff=FIR_BAND_HZ, fs=Fs, pass_zero=False)

    pn_signal = filtfilt(b=b, a=[1], x=make_pink_noise(PINK_NOISE_ALPHA, n_samp, 1 / Fs))
    pn_noise = make_pink_noise(PINK_NOISE_ALPHA, n_samp, 1 / Fs)

    Vlo = A * (pn_signal / pn_signal.std())
    data = Vlo + PINK_NOISE_SNR * pn_noise
    gt_states: npt.NDArray[np.complex_] = hilbert(Vlo)  # pyright: ignore
    true_phase = np.angle(gt_states)
    return SimulationResults(data, gt_states, true_phase)


def gen_state_space_model_white(duration_sec: float, Fs: float) -> SimulationResults:
    SIGNAL_SIGMA_GT = np.sqrt(10)
    NOISE_SIGMA_GT = 1

    mp = MatsudaParams(A=0.99, freq=6, sr=Fs)
    oscillation_model = SingleRhythmModel(mp, sigma=SIGNAL_SIGMA_GT)
    gt_states = collect(oscillation_model, int(duration_sec * Fs))

    noise = NOISE_SIGMA_GT * np.random.randn(int(duration_sec * Fs))
    data: npt.NDArray[np.floating] = np.real(gt_states) + noise  # type: ignore
    true_phase = np.angle(gt_states)  # type: ignore
    return SimulationResults(data, gt_states, true_phase)  # type: ignore


def gen_state_space_model_pink(duration_sec: float, Fs: float) -> SimulationResults:
    SIGNAL_SIGMA_GT = np.sqrt(10)
    NOISE_SIGMA_GT = 1
    ALPHA = 1.5
    NOISE_AR_ORDER = 1000

    mp = MatsudaParams(A=0.99, freq=6, sr=Fs)
    oscillation_model = SingleRhythmModel(mp, sigma=SIGNAL_SIGMA_GT)
    gt_states = collect(oscillation_model, int(duration_sec * Fs))

    x0 = np.random.rand(NOISE_AR_ORDER)
    noise_model = ArNoise(x0=x0, alpha=ALPHA, order=NOISE_AR_ORDER, s=NOISE_SIGMA_GT)
    noise = collect(noise_model, int(duration_sec * Fs))

    data: npt.NDArray[np.floating] = np.real(gt_states) + noise  # type: ignore
    true_phase = np.angle(gt_states)  # type: ignore
    return SimulationResults(data, gt_states, true_phase)  # type: ignore


def gen_two_sines(duration_sec: float, Fs: float, a2: float, f2: float) -> SimulationResults:
    """
    Parameters
    ----------
    duration_sec : float
        Generated signal duration in seconds
    Fs : float
        Sampling frequency
    a2 : float
        Amplitude scaling factor for the second (confounding) oscillation
    f2 : float
        Frequency of the confounding oscillation

    """
    FREQ_HZ = 6
    A1, D_PHI1, SNR = 25, np.pi / 3, 0.5

    n_samp = int(duration_sec * Fs)
    times = np.arange(1, n_samp + 1) / Fs

    true_phase = 2 * np.pi * FREQ_HZ * times
    sine1 = A1 * np.cos(true_phase)
    sine2 = a2 * A1 * np.cos(2 * np.pi * f2 * times + D_PHI1)
    data = sine1 + sine2 + SNR * np.random.randn(n_samp)
    true_phase = _wrapToPi(true_phase)
    gt_states = sine1 + 1j * A1 * np.sin(true_phase)
    return SimulationResults(data, gt_states, true_phase)


def gen_phase_reset_data(duration_sec: float, Fs: float) -> SimulationResults:
    FREQ_HZ = 6
    RESET_SAMPS = [3500, 4750, 6500, 8500]
    A, SNR = 25, 10
    PINK_NOISE_ALPHA = 1.5

    n_samp = int(duration_sec * Fs)
    noise = make_pink_noise(PINK_NOISE_ALPHA, n_samp, 1 / Fs)
    # Make the data.
    times = np.arange(1, n_samp + 1) / Fs
    omega = 2 * np.pi * FREQ_HZ
    V1 = A * np.cos(omega * times[: RESET_SAMPS[0]])
    V2 = A * np.cos(omega * times[RESET_SAMPS[0] : RESET_SAMPS[1]] + np.pi / 2)
    V3 = A * np.cos(omega * times[RESET_SAMPS[1] : RESET_SAMPS[2]])
    V4 = A * np.cos(omega * times[RESET_SAMPS[2] : RESET_SAMPS[3]] + np.pi / 2)
    V5 = A * np.cos(omega * times[RESET_SAMPS[3] :])

    Vlo = np.concatenate([V1, V2, V3, V4, V5])
    data = Vlo + SNR * noise

    # truePhase = wrapTo2Pi(
    true_phase = _wrapToPi(
        np.concatenate(
            [
                omega * times[: RESET_SAMPS[0]],
                omega * times[RESET_SAMPS[0] : RESET_SAMPS[1]] + np.pi / 2,
                omega * times[RESET_SAMPS[1] : RESET_SAMPS[2]],
                omega * times[RESET_SAMPS[2] : RESET_SAMPS[3]] + np.pi / 2,
                omega * times[RESET_SAMPS[3] :],
            ]
        )
    )
    gt_states = np.concatenate(
        [
            V1 + 1j * A * np.sin(omega * times[: RESET_SAMPS[0]]),
            V2 + 1j * A * np.sin(omega * times[RESET_SAMPS[0] : RESET_SAMPS[1]] + np.pi / 2),
            V3 + 1j * A * np.sin(omega * times[RESET_SAMPS[1] : RESET_SAMPS[2]]),
            V4 + 1j * A * np.sin(omega * times[RESET_SAMPS[2] : RESET_SAMPS[3]] + np.pi / 2),
            V5 + 1j * A * np.sin(omega * times[RESET_SAMPS[3] :]),
        ]
    )
    return SimulationResults(data, gt_states, true_phase)


def _wrapToPi(phase: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Emulate MATLAB's wrapToPi, https://stackoverflow.com/a/71914752"""
    xwrap = np.remainder(phase, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = phase < 0
    mask2 = np.remainder(phase, np.pi) == 0
    mask3 = np.remainder(phase, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap


if __name__ == "__main__":
    import doctest

    doctest.testmod()
