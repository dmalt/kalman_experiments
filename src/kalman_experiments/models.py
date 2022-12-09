from __future__ import annotations

from cmath import exp
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from mne.io.brainvision.brainvision import read_raw_brainvision  # type: ignore

from kalman_experiments.numpy_types import Timeseries, Vec1D

from .complex import complex_randn


class SignalGenerator(Protocol):
    def step(self) -> complex:
        """Generate single noise sample"""
        ...


@dataclass
class MatsudaParams:
    """Single oscillation Matsuda-Komaki model parameters"""
    A: float
    freq: float
    sr: float

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq / self.sr * 1j)


@dataclass
class SingleRhythmModel:
    mp: MatsudaParams
    cont_sigma: float
    x: complex = 0

    def step(self) -> complex:
        """Update model state and generate measurement"""
        sigma_discrete = self.cont_sigma * np.sqrt(self.mp.sr)
        self.x = self.mp.Phi * self.x + complex_randn() * sigma_discrete
        return self.x

    def psd_onesided(self, f: float) -> float:
        """
        Theoretical PSD for Matsuda-Komaki multivariate AR process

        Notes
        -----
        Implementation follows eq. (6.38) from [1] for the first state component, i.e.
        it effectively computes p_{11} from (6.38) for the single-rhythm MK model

        N.B.: eq. (6.38) is for two-sided spectrum. To match the default of scipy.signal.welch,
        we return onesided spectrum, which is double the original

        References
        ----------
        .. [1] Kitagawa, Genshiro. 2010. Introduction to Time Series Modeling.
        0 ed. Chapman and Hall/CRC. https://doi.org/10.1201/9781584889229.

        """
        phi = 2 * np.pi * self.mp.freq / self.mp.sr
        psi = 2 * np.pi * f / self.mp.sr
        A = self.mp.A

        denom = np.abs(1 - 2 * A * np.cos(phi) * np.exp(-1j * psi) + A**2 * np.exp(-2j * psi)) ** 2
        num = 1 + A**2 - 2 * A * np.cos(phi) * np.cos(psi)
        return self.cont_sigma**2 * num / denom * 2


def gen_ar_noise_coefficients(alpha: float, order: int) -> Vec1D:
    """
    Parameters
    ----------
    order : int
        Order of the AR model
    alpha : float in the [-2, 2] range
        Alpha as in '1/f^alpha' PSD profile

    References
    ----------
    .. [1] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings
    of the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.

    """
    a: list[float] = [1]
    for k in range(1, order + 1):
        a.append((k - 1 - alpha / 2) * a[-1] / k)  # AR coefficients as in [1], eq. (116)
    return -np.array(a[1:])


class ArNoiseModel:
    """
    Generate 1/f^alpha noise with truncated autoregressive process, as described in [1]

    Parameters
    ----------
    x0 : np.ndarray of shape(order,)
        Initial conditions vector for the AR model
    order : int
        Order of the AR model
    alpha : float in range [-2, 2]
        Alpha as in '1/f^alpha'
    s : float, >= 0
        White noise standard deviation (see [1])

    References
    ----------
    .. [1] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings
    of the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.

    """

    def __init__(self, x0: np.ndarray, order: int = 1, alpha: float = 1, s: float = 1):
        assert (len(x0) == order), f"x0 length must match AR order; got {len(x0)=}, {order=}"
        self.a = gen_ar_noise_coefficients(alpha, order)
        self.x = x0
        self.s = s

    def step(self) -> float:
        """Make one step of the AR process"""
        y_next = self.a @ self.x + np.random.randn() * self.s
        self.x = np.concatenate([[y_next], self.x[:-1]])  # type: ignore
        return float(y_next)


class RealNoise:
    def __init__(self, single_channel_eeg: Timeseries, s: float):
        self.single_channel_eeg = single_channel_eeg
        self.ind = 0
        self.s = s

    def step(self) -> float:
        n_samp = len(self.single_channel_eeg)
        if self.ind >= len(self.single_channel_eeg):
            raise IndexError(f"Index {self.ind} is out of bounds for data of length {n_samp}")
        self.ind += 1
        return self.single_channel_eeg[self.ind] * self.s


def prepare_real_noise(
    raw_path: str, s: float = 1, minsamp: int = 0, maxsamp: int | None = None
) -> tuple[RealNoise, float]:
    raw = read_raw_brainvision(raw_path, preload=True, verbose="ERROR")
    raw.pick_channels(["FC2"])
    raw.crop(tmax=244)
    raw.filter(l_freq=0.1, h_freq=None, verbose="ERROR")

    data = np.squeeze(raw.get_data())
    data /= data.std()
    data -= data.mean()
    crop = slice(minsamp, maxsamp)
    return RealNoise(data[crop], s), raw.info["sfreq"]


def collect(signal_generator: SignalGenerator, n_samp: int) -> Timeseries:
    return np.array([signal_generator.step() for _ in range(n_samp)])
