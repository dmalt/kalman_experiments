from __future__ import annotations

import numpy as np


def plv(x1, x2, ma_window_samp: int):
    x1, x2 = x1.copy(), x2.copy()
    x1 /= np.abs(x1)
    x2 /= np.abs(x2)
    prod = np.conj(x1) * x2

    ma_kernel = np.ones(ma_window_samp) / ma_window_samp
    assert ma_window_samp > 0
    return np.convolve(prod, ma_kernel, mode="same"), prod[ma_window_samp:-ma_window_samp].mean()


def env_cor(x1, x2):
    return np.corrcoef(np.abs(x1), np.abs(x2))[0, 1]


def crosscorr(
    c1: np.ndarray, c2: np.ndarray, sr: float, shift_nsamp: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = c1.copy(), c2[shift_nsamp:-shift_nsamp].copy()
    c1 -= c1.mean()
    c2 -= c2.mean()
    cc1, cc2 = np.correlate(c1, c1), np.correlate(c2, c2)
    times = np.arange(-shift_nsamp, shift_nsamp + 1) / sr
    return times, np.correlate(c2, c1) / np.sqrt(cc1 * cc2)
