from __future__ import annotations

import random
from cmath import exp
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


class ArNoiseGenerator(Protocol):
    def step(self) -> float:
        ...


def complex_randn() -> complex:
    return random.gauss(0, 1) + 1j * random.gauss(0, 1)


V2 = npt.NDArray  # array of shape(2,)
M2 = npt.NDArray  # array of shape(2, 2)


def complex2vec(x: complex) -> V2:
    return np.array([x.real, x.imag])


def vec2complex(x: V2) -> complex:
    return x[0] + 1j * x[1]


def complex2mat(X: complex) -> M2:
    return np.array([[X.real, -X.imag], [X.imag, X.real]])


@dataclass
class SingleRhythmModel:
    freq: float
    A: float
    sigma: float
    sr: float
    meas_noise_ar: ArNoiseGenerator
    x: complex = 0

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq / self.sr * 1j)

    def step(self) -> float:
        """Update model state and generate measurement"""
        self.x = self.Phi * self.x + complex_randn() * self.sigma
        return self.x.real + self.meas_noise_ar.step()


if __name__ == "__main__":
    from gen_ar_noise import ArNoise
    from real_noise import prepare_real_noise

    order = 1
    alpha = 1
    meas_noise = ArNoise(y0=np.random.rand(order), alpha=alpha, order=order, sigma=1.5)
    # meas_noise = prepare_real_noise(raw_path="./sub-01_ses-session2_task-eyesopen_eeg.vhdr")
    sr = 125
    model = SingleRhythmModel(freq=15, A=0.99, sigma=1, sr=sr, meas_noise_ar=meas_noise)

    res = []
    for i in range(100_000):
        res.append(model.step())
        # res.append(meas_noise.step())
    res = np.array(res)

    import matplotlib.pyplot as plt
    from scipy.signal import welch

    freqs, psd = welch(res, fs=sr, nperseg=1024)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    freq_lim = 1000
    ax1.plot(freqs[1:freq_lim], psd[1:freq_lim])
    ax1.plot(freqs[1:freq_lim], [1 / f**alpha - 0.001 for f in freqs[1:freq_lim]])
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    # ax1.vlines(10, ymin=0, ymax=10)
    ax1.legend([f"AR({order}) for 1/f noise", "1/f"])
    ax1.set_xlabel("Frequencies, Hz")
    ax1.grid()
    ax2.plot(np.linspace(0, 2, 1000), res[:1000])
    ax2.set_xlabel("Time, sec")

    plt.show()
