import numpy as np
from mne.io import read_raw_brainvision

from kalman_experiments.numpy_types import Vec


def gen_ar_noise_coefficients(alpha: float, order: int) -> Vec:
    """
    Parameters
    ----------
    order : int
        Order of the AR model
    alpha : float in range [-2, 2]
        Alpha as in '1/f^alpha'

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
    order : int
        Order of the AR model
    alpha : float in range [-2, 2]
        Alpha as in '1/f^alpha'
    sigma : float, >= 0
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
    def __init__(self, single_channel_eeg: np.ndarray):
        self.single_channel_eeg = single_channel_eeg
        self.ind = 0

    def step(self) -> float:
        n_samp = len(self.single_channel_eeg)
        if self.ind >= len(self.single_channel_eeg):
            raise IndexError(f"Index {self.ind} is out of bounds for data of length {n_samp}")
        self.ind += 1
        return self.single_channel_eeg[self.ind]


def prepare_real_noise(
    raw_path: str, sigma: float = 1, minsamp: int = 0, maxsamp: int | None = None
) -> tuple[RealNoise, float]:
    raw = read_raw_brainvision(raw_path, preload=True)
    raw.pick_channels(["FC2"])
    raw.crop(tmax=244)
    raw.filter(l_freq=0.1, h_freq=None)

    data = np.squeeze(raw.get_data())
    data /= data.std()
    data -= data.mean()
    print("dm=", data.mean())
    data *= sigma
    crop = slice(minsamp, maxsamp)
    return RealNoise(data[crop]), raw.info["sfreq"]
