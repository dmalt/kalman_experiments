import numpy as np


def plv(x1, x2, ma_len_samp):
    x1 /= np.abs(x1)
    x2 /= np.abs(x2)
    prod = np.conj(x1) * x2

    ma_kernel = np.ones(ma_len_samp) / ma_len_samp
    return np.convolve(prod, ma_kernel, mode="same"), prod.mean()


def env_cor(x1, x2):
    return np.corrcoef(np.abs(x1), np.abs(x2))[0, 1]


def crosscor(x1, x2):
    return np.correlate(x1.real, x2.real)
