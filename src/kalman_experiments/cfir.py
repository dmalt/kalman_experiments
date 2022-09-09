from dataclasses import asdict, dataclass

import numpy as np
import scipy.signal as sg

from .numpy_types import Vec1D


class CFIRBandDetector:
    """
    Complex-valued FIR envelope detector based on analytic signal reconstruction

    Parameters
    ----------
    band : tuple[float, float]
        Frequency range to apply band-pass filtering
    sr : float
        Sampling frequency
    delay : int
        Delay of ideal filter in samples
    n_taps : positive int
        Length of FIR filter
    n_fft : positive int
        Length of frequency grid to estimate ideal freq. response
    weights : array of shape(n_weights,) or None
        Least squares weights. If None match WHilbertFilter

    """

    def __init__(
        self,
        band: tuple[float, float],
        sr: float,
        delay: int,
        n_taps: int = 500,
        n_fft: int = 2000,
        weights: Vec1D | None = None,
    ):
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * sr < band[0]) | (w / n_fft * sr > band[1])] = 0
        F = np.array(
            [np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)]
        )
        if weights is None:
            self.b = F.T.conj().dot(H) / n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.0])
        self.zi = np.zeros(len(self.b) - 1)

    def apply(self, signal: Vec1D) -> Vec1D:
        y, self.zi = sg.lfilter(self.b, self.a, signal, zi=self.zi)
        return y


@dataclass
class CFIRParams:
    band: tuple[float, float]
    sr: float
    n_taps: int = 500
    n_fft: int = 2000
    weights: Vec1D | None = None


def apply_cfir(cfir_params: CFIRParams, signal: Vec1D, delay: int) -> Vec1D:
    cfir_params_dict = asdict(cfir_params) | {"delay": delay}
    cfir = CFIRBandDetector(**cfir_params_dict)
    x, cfir.zi = sg.lfilter(cfir.b, cfir.a, signal, zi=cfir.zi)
    return x
