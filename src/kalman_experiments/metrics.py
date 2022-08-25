import numpy as np


def plv(x1, x2, ma_len_samp):
    x1 /= np.abs(x1)
    x2 /= np.abs(x2)
    prod = np.conj(x1) * x2

    ma_kernel = np.ones(ma_len_samp) / ma_len_samp
    return np.convolve(prod, ma_kernel, mode="same"), prod[: len(prod) // 2].mean()


def env_cor(x1, x2, ma_len_samp):
    x1 = np.abs(x1)
    x1 -= x1.mean()
    x1 /= x1.std()
    x2 = np.abs(x2)
    x2 -= x2.mean()
    x2 /= x2.std()
    prod = x1 * x2

    ma_kernel = np.ones(ma_len_samp) / ma_len_samp
    return np.convolve(prod, ma_kernel, mode="same"), prod[: len(prod) // 2].mean()
