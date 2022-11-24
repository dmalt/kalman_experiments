from __future__ import annotations

from cmath import exp
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from mne.io.brainvision.brainvision import read_raw_brainvision  # type: ignore

from kalman_experiments.numpy_types import (
    Alpha,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Timeseries,
    Vec,
    check_in_alpha_range,
    check_nonnegative_float,
    check_nonnegative_int,
    check_positive_float,
    check_positive_int,
)

from .complex import complex_randn


class SignalGenerator(Protocol):
    def step(self) -> complex:
        """Generate single noise sample"""
        ...


@dataclass
class MatsudaParams:
    """Single oscillation Matsuda-Komaki model parameters"""

    A: NonNegativeFloat
    freq_hz: NonNegativeFloat
    sr: PositiveFloat

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq_hz / self.sr * 1j)


@dataclass
class SingleRhythmModel:
    mp: MatsudaParams
    sigma: NonNegativeFloat
    x: complex = 0

    def step(self) -> complex:
        """Update model state and generate measurement"""
        self.x = self.mp.Phi * self.x + complex_randn() * self.sigma
        return self.x


def gen_ar_noise_coefficients(alpha: Alpha, order: PositiveInt) -> Vec:
    """
    Parameters
    ----------
    order : positive int
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
        a.append((k - 1 - alpha / 2) * a[-1] / k)  # AR coefficients as in [1]
    return -np.array(a[1:])


class ArNoise:
    """
    Generate 1/f^alpha noise with truncated autoregressive process, as described in [1]

    Parameters
    ----------
    x0 : np.ndarray of shape(order,)
        Initial conditions vector for the AR model
    order : int, > 0
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

    def __init__(
        self,
        x0: np.ndarray,
        order: PositiveInt = check_positive_int(1),
        alpha: Alpha = check_in_alpha_range(1),
        s: NonNegativeFloat = check_positive_float(1),
    ):
        assert len(x0) == order, f"x0 length must match AR order; got {len(x0)=}, {order=}"
        self.a = gen_ar_noise_coefficients(alpha, order)
        self.x = x0
        self.s = s

    def step(self) -> float:
        """Make one step of the AR process"""
        y_next = self.a @ self.x + np.random.randn() * self.s
        self.x = np.concatenate([[y_next], self.x[:-1]])  # type: ignore
        return float(y_next)


class RealNoise:
    def __init__(self, single_channel_eeg: Timeseries, s: NonNegativeFloat):
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
    raw_path: str,
    s: NonNegativeFloat = check_nonnegative_float(1),
    minsamp: NonNegativeInt = check_nonnegative_int(0),
    maxsamp: PositiveInt | None = None,
) -> tuple[RealNoise, NonNegativeFloat]:
    raw = read_raw_brainvision(raw_path, preload=True, verbose="ERROR")
    raw.pick_channels(["FC2"])
    raw.crop(tmax=244)
    raw.filter(l_freq=0.1, h_freq=None, verbose="ERROR")

    data = np.squeeze(raw.get_data())
    data /= data.std()
    data -= data.mean()
    crop = slice(minsamp, maxsamp)
    return RealNoise(data[crop], s), check_positive_float(raw.info["sfreq"])


def collect(signal_generator: SignalGenerator, n_samp: PositiveInt) -> Timeseries:
    return np.array([signal_generator.step() for _ in range(n_samp)])
