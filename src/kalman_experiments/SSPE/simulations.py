import numpy as np
from scipy.signal import filtfilt, firwin, hilbert

from .make_pink_noise import make_pink_noise

FREQ_HZ = 6
PINK_NOISE_ALPHA = 1.5
SIN_AMP = 10
PINK_NOISE_SNR = 10


def generate_sines_w_pink(duration_sec, Fs):
    noise = make_pink_noise(PINK_NOISE_ALPHA, duration_sec * Fs, 1 / Fs)
    times = np.arange(1, duration_sec * Fs + 1) / Fs
    Vlo = SIN_AMP * np.cos(2 * np.pi * FREQ_HZ * times)
    data = Vlo + PINK_NOISE_SNR * noise
    true_phase = wrapToPi(2 * np.pi * FREQ_HZ * times)
    return data, true_phase


def generate_sines_w_white(duration_sec, Fs):
    times = np.arange(1, duration_sec * Fs + 1) / Fs
    Vlo = SIN_AMP * np.cos(2 * np.pi * FREQ_HZ * times)
    data = Vlo + np.random.randn(len(Vlo))
    true_phase = wrapToPi(2 * np.pi * FREQ_HZ * times)
    return data, true_phase


def wrapToPi(x):
    """Emulate MATLAB's wrapToPi, https://stackoverflow.com/a/71914752"""
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap


def generate_two_sines(duration_sec, Fs, second_amp_factor, second_freq):
    times = np.arange(1, duration_sec * Fs + 1) / Fs
    data = (
        25 * np.cos(2 * np.pi * FREQ_HZ * times)
        + second_amp_factor * 25 * np.cos(2 * np.pi * second_freq * times + np.pi / 3)
        + 0.5 * np.random.randn(len(times))
    )
    true_phase = wrapToPi(2 * np.pi * FREQ_HZ * times)
    return data, true_phase


def generate_phase_reset_data(duration_sec, Fs, time_points_splice=[3500, 4750, 6500, 8500]):
    # [pn] = make_pink_noise(1.5,time*Fs,1/Fs);
    noise = make_pink_noise(PINK_NOISE_ALPHA, duration_sec * Fs, 1 / Fs)
    # % Make the data.
    times = np.arange(1, duration_sec * Fs + 1) / Fs
    omega = 2 * np.pi * FREQ_HZ
    V1 = 25 * np.cos(omega * times[: time_points_splice[0]])
    V2 = 25 * np.cos(omega * times[time_points_splice[0] : time_points_splice[1]])
    V3 = 25 * np.cos(omega * times[time_points_splice[1] : time_points_splice[2]])
    V4 = 25 * np.cos(omega * times[time_points_splice[2] : time_points_splice[3]])
    V5 = 25 * np.cos(omega * times[time_points_splice[3] :])

    Vlo = np.concatenate([V1, V2, V3, V4, V5])
    data = Vlo + 10 * noise

    # truePhase = wrapTo2Pi(
    true_phase = wrapToPi(
        np.concatenate(
            [
                omega * times[: time_points_splice[0]],
                omega * times[time_points_splice[0] : time_points_splice[1]] + np.pi / 2,
                omega * times[time_points_splice[1] : time_points_splice[2]],
                omega * times[time_points_splice[2] : time_points_splice[3]] + np.pi / 2,
                omega * times[time_points_splice[3] :],
            ]
        )
    )
    return data, true_phase


def generate_filtered_pink_noise_with_added_pink_noise(duration_sec, Fs):
    ORDER = 750
    # D = designfilt('bandpassfir', 'FilterOrder', default_parameters.filter_order, 'CutoffFrequency1', 4, 'CutoffFrequency2', 8, 'SampleRate', Fs, 'DesignMethod', 'window');
    BAND_HZ = [4, 8]
    b = firwin(numtaps=ORDER + 1, cutoff=BAND_HZ, fs=Fs, pass_zero=False)

    # import matplotlib.pyplot as plt
    # from scipy.signal import freqz
    # w, h = freqz(b, 1, worN=2000)
    # plt.plot((Fs * 0.5 / np.pi) * w, abs(h), label="Hamming window")
    # plt.show()
    pn_signal = make_pink_noise(PINK_NOISE_ALPHA, duration_sec * Fs, 1 / Fs)
    # pn_signal = np.random.randn(duration_sec * Fs)
    pn_noise = make_pink_noise(PINK_NOISE_ALPHA, duration_sec * Fs, 1 / Fs)
    pn_signal = filtfilt(b=b, a=[1], x=pn_signal)

    Vlo = 10 * (pn_signal / pn_signal.std())
    data = Vlo + PINK_NOISE_SNR * pn_noise
    true_phase = np.angle(hilbert(Vlo))  # type: ignore
    return data, true_phase



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    # data, true_phase = generate_sines_w_pink(10, 1000)
    # data, true_phase = generate_sines_w_white(10, 1000)
    # data, true_phase = generate_phase_reset_data(10, 1000)
    data, true_phase = generate_filtered_pink_noise_with_added_pink_noise(10, 1000)
    freqs, psd_noise = welch(data, fs=1000, nperseg=1000)
    # print(freqs)
    # plt.plot(data)
    plt.plot(freqs[:51], 10 * np.log10(psd_noise)[:51])
    # plt.gca().set_xscale("log")
    plt.grid()
    # plt.plot(true_phase)
    plt.show()
