"""
Simulations from SSPE paper

Examples
--------
Generate 10 seconds of data sampled at 1000 Hz and plot the PSD
>>> import matplotlib.pyplot as plt
>>> from scipy.signal import welch
>>> duration, fs, freq_lim = 10, 1000, 51
>>> data_wn, true_phase_wn = gen_sine_w_white(duration, fs)
>>> data_pn, true_phase_pn = gen_sine_w_pink(duration, fs)
>>> # broadband oscillation in pink noise
>>> data_pn_bb, true_phase_pn_bb = gen_filt_pink_noise_w_added_pink_noise(duration, fs)
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
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt, firwin, hilbert


def make_pink_noise(alpha: float, n_samp: int, dt: float) -> npt.NDArray[np.floating[Any]]:
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
    data: npt.NDArray[np.floating[Any]]
    true_phase: npt.NDArray[np.floating[Any]]


def gen_sine_w_white(duration_sec: float, Fs: float) -> SimulationResults:
    """Generate sine in white noise"""
    FREQ_HZ = 6
    A = 10

    n_samp = int(duration_sec * Fs)
    times = np.arange(1, n_samp + 1) / Fs
    Vlo = A * np.cos(2 * np.pi * FREQ_HZ * times)
    data = Vlo + np.random.randn(n_samp)
    true_phase = _wrapToPi(2 * np.pi * FREQ_HZ * times)
    return SimulationResults(data, true_phase)


def gen_sine_w_pink(duration_sec: float, Fs: float) -> SimulationResults:
    """Generate sine in pink noise"""
    FREQ_HZ = 6
    A = 10
    PINK_NOISE_SNR = 10
    PINK_NOISE_ALPHA = 1.5

    n_samp = int(duration_sec * Fs)
    noise = make_pink_noise(PINK_NOISE_ALPHA, n_samp, 1 / Fs)
    times = np.arange(1, n_samp + 1) / Fs
    Vlo = A * np.cos(2 * np.pi * FREQ_HZ * times)
    data = Vlo + PINK_NOISE_SNR * noise
    true_phase = _wrapToPi(2 * np.pi * FREQ_HZ * times)
    return SimulationResults(data, true_phase)


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
    true_phase = np.angle(hilbert(Vlo))  # type: ignore
    return SimulationResults(data, true_phase)


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
    return SimulationResults(data, true_phase)


def gen_phase_reset_data(duration_sec: float, Fs: float) -> SimulationResults:
    FREQ_HZ = 6
    TIME_POINTS_SPLICE = [3500, 4750, 6500, 8500]
    A, SNR = 25, 10
    PINK_NOISE_ALPHA = 1.5

    n_samp = int(duration_sec * Fs)
    noise = make_pink_noise(PINK_NOISE_ALPHA, n_samp, 1 / Fs)
    # Make the data.
    times = np.arange(1, n_samp + 1) / Fs
    omega = 2 * np.pi * FREQ_HZ
    V1 = A * np.cos(omega * times[: TIME_POINTS_SPLICE[0]])
    V2 = A * np.cos(omega * times[TIME_POINTS_SPLICE[0] : TIME_POINTS_SPLICE[1]])
    V3 = A * np.cos(omega * times[TIME_POINTS_SPLICE[1] : TIME_POINTS_SPLICE[2]])
    V4 = A * np.cos(omega * times[TIME_POINTS_SPLICE[2] : TIME_POINTS_SPLICE[3]])
    V5 = A * np.cos(omega * times[TIME_POINTS_SPLICE[3] :])

    Vlo = np.concatenate([V1, V2, V3, V4, V5])
    data = Vlo + SNR * noise

    # truePhase = wrapTo2Pi(
    true_phase = _wrapToPi(
        np.concatenate(
            [
                omega * times[: TIME_POINTS_SPLICE[0]],
                omega * times[TIME_POINTS_SPLICE[0] : TIME_POINTS_SPLICE[1]] + np.pi / 2,
                omega * times[TIME_POINTS_SPLICE[1] : TIME_POINTS_SPLICE[2]],
                omega * times[TIME_POINTS_SPLICE[2] : TIME_POINTS_SPLICE[3]] + np.pi / 2,
                omega * times[TIME_POINTS_SPLICE[3] :],
            ]
        )
    )
    return SimulationResults(data, true_phase)


def _wrapToPi(phase: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
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
