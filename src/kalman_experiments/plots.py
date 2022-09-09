import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, welch


def plot_generated_signal(noise, meas, sr, alpha, legend, tmin=0, tmax=2):
    freqs, psd_noise = welch(noise, fs=sr, nperseg=1024)
    freqs, psd_signal = welch(meas, fs=sr, nperseg=1024)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    freq_lim = 1000
    one_over_f = np.array([1 / f**alpha for f in freqs[1:freq_lim]])

    # bring 1/f line closer to data
    one_over_f *= psd_noise[min(len(psd_noise), freq_lim) - 1] / one_over_f[-1]

    ax1.loglog(freqs[1:freq_lim], psd_signal[1:freq_lim])
    ax1.loglog(freqs[1:freq_lim], one_over_f)
    ax1.loglog(freqs[1:freq_lim], psd_noise[1:freq_lim], alpha=0.5)

    ax1.legend(legend)
    ax1.set_xlabel("Frequencies, Hz")
    ax1.grid()
    t = np.linspace(tmin, tmax, (tmax - tmin) * sr, endpoint=False)
    ax2.plot(t, meas[int(tmin * sr) : int(tmax * sr)])
    ax2.grid()
    ax2.set_xlabel("Time, sec")

    return fig, ax1, ax2


def plot_kalman_vs_cfir(
    meas, gt_states, kf_states, cfir_states, plv_win_kf, plv_win_cfir, n_samp, sr, delay
):
    t = np.arange(n_samp) / sr
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
    ax1.plot(t, np.roll(np.real(kf_states), shift=-delay)[2 * n_samp : 3 * n_samp], alpha=0.9)
    ax1.plot(t, np.roll(np.real(cfir_states), shift=-delay)[2 * n_samp : 3 * n_samp], alpha=0.9)
    ax1.plot(t, meas[2 * n_samp : 3 * n_samp], alpha=0.3, linewidth=1)
    ax1.plot(t, np.real(gt_states)[2 * n_samp : 3 * n_samp], "-", alpha=0.3, linewidth=4)
    ax1.legend(["kalman state (Re)", "cfir state (Re)", "measurements", "ground truth state (Re)"])
    ax1.grid()

    plt.xlabel("Time, sec")
    ax2.plot(t, np.abs(plv_win_kf)[2 * n_samp : 3 * n_samp], linewidth=2)
    ax2.plot(t, np.abs(plv_win_cfir)[2 * n_samp : 3 * n_samp], linewidth=2)
    ax2.legend(["plv(gt, kf)", "plv(gt, cfir)"])
    ax2.grid()

    ax3.plot(t, np.abs(kf_states)[2 * n_samp : 3 * n_samp], alpha=0.9)
    ax3.plot(t, np.abs(cfir_states)[2 * n_samp : 3 * n_samp], alpha=0.7)
    ax3.plot(t, np.abs(gt_states)[2 * n_samp : 3 * n_samp], alpha=0.7)
    ax3.plot(t, np.abs(hilbert(meas))[2 * n_samp : 3 * n_samp], alpha=0.3)  # pyright: ignore
    ax3.legend(["kalman envelope", "cfir envelope", "gt envelope", "meas envelope"])
    ax3.grid()
    return f, ax1, ax2, ax3


def plot_crosscorrelations(t_ms, corrs_cfir, corrs_kf):
    fig = plt.figure(figsize=(9, 5))
    ax = plt.subplot()
    ind_cfir = np.argmax(corrs_cfir)
    ind_kf = np.argmax(corrs_kf)

    ax.plot(t_ms, corrs_cfir, "C1", label="cfir")
    ax.axvline(t_ms[ind_cfir], color="C1")
    ax.axhline(corrs_cfir[ind_cfir], color="C1")
    ax.plot(t_ms, corrs_kf, "C2", label="kalman")
    ax.axvline(t_ms[ind_kf], color="C2")
    ax.axhline(corrs_kf[ind_kf], color="C2")
    ax.set_xlabel("delay, ms")
    ax.set_ylabel("correlation")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid()

    ax.annotate(
        f"d={t_ms[ind_kf]} ms, c={round(corrs_kf[ind_kf], 2)}",
        (t_ms[ind_kf] + 0.002, corrs_kf[ind_kf] + 0.02),
    )
    ax.annotate(
        f"d={t_ms[ind_cfir]} ms, c={round(corrs_cfir[ind_cfir], 2)}",
        (t_ms[ind_cfir] + 0.002, corrs_cfir[ind_cfir] - 0.05),
    )

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    return fig, ax
