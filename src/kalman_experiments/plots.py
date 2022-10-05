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


def plot_timeseries(ax, times, timeseries: list[dict[str, Any]]):
    for ts in timeseries:
        ax.plot(times, **ts)
    ax.legend()
    ax.grid()


def plot_kalman_vs_cfir(
    meas, gt_states, kf_states, cfir_states, plv_win_kf, plv_win_cfir, n_samp, sr, delay
):
    times = np.arange(n_samp) / sr
    meas = meas[2 * n_samp : 3 * n_samp]
    gt_states = gt_states[2 * n_samp : 3 * n_samp]
    kf_states = np.roll(kf_states, shift=-delay)[2 * n_samp : 3 * n_samp]
    cfir_states = np.roll(cfir_states, shift=-delay)[2 * n_samp : 3 * n_samp]

    plv_win_kf = plv_win_kf[2 * n_samp : 3 * n_samp]
    plv_win_cfir = plv_win_cfir[2 * n_samp : 3 * n_samp]

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
    plot_timeseries(
        ax1,
        times,
        [
            dict(y=np.real(kf_states), alpha=0.9, label="kalman state (Re)"),
            dict(y=np.real(cfir_states), alpha=0.9, label="cfir state (Re)"),
            dict(y=meas, alpha=0.3, linewidth=1, label="measurements"),
            dict(y=np.real(gt_states), alpha=0.3, linewidth=4, label="ground truth state (Re)"),
        ],
    )

    plot_timeseries(
        ax2,
        times,
        [
            dict(y=np.abs(plv_win_kf), linewidth=2, label="plv(gt, kf)"),
            dict(y=np.abs(plv_win_cfir), linewidth=2, label="plv(gt, cfir)"),
        ],
    )

    plot_timeseries(
        ax3,
        times,
        [
            dict(y=np.abs(kf_states), alpha=0.9, label="kalman envelope"),
            dict(y=np.abs(cfir_states), alpha=0.7, label="cfir envelope"),
            dict(y=np.abs(gt_states), alpha=0.7, label="gt envelope"),
            dict(y=np.abs(hilbert(meas)), alpha=0.3, label="meas envelope"),  # pyright: ignore
        ]
    )
    plt.xlabel("Time, sec")
    return f, ax1, ax2, ax3


def plot_crosscorrelations(t_ms, corrs_cfir, corrs_kf):
    fig = plt.figure(figsize=(9, 5))
    # fig = plt.figure()
    ax = plt.subplot()
    ind_cfir = np.argmax(corrs_cfir)
    ind_kf = np.argmax(corrs_kf)

    C1 = "#d1a683"
    C2 = "#005960"
    ax.plot(t_ms, corrs_cfir, color=C1, label="CFIR")
    ax.axvline(t_ms[ind_cfir], color=C1)
    ax.axhline(corrs_cfir[ind_cfir], color=C1)
    ax.plot(t_ms, corrs_kf, C2, label="Kalman")
    ax.axvline(t_ms[ind_kf], color=C2)
    ax.axhline(corrs_kf[ind_kf], color=C2)
    ax.set_xlabel("delay, ms", fontsize=14)
    ax.set_ylabel("correlation", fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=14)
    ax.grid()

    ax.annotate(f"{t_ms[ind_kf]} ms", (t_ms[ind_kf] + 1, 0.02), color=C2, fontsize=16)
    ax.annotate(f"{t_ms[ind_cfir]} ms", (t_ms[ind_cfir] + 1, 0.02), color=C1, fontsize=16)
    ax.annotate(f"{corrs_kf[ind_kf]:.2f}", (-100, corrs_kf[ind_kf] + 0.01), color=C2, fontsize=16)
    ax.annotate(
        f"{corrs_cfir[ind_cfir]:.2f}",
        (-100, corrs_cfir[ind_cfir] + 0.01),
        color=C1,
        fontsize=16,
    )

    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    return fig, ax
