import numpy as np
import scipy.signal as sg


class CFIRBandDetector:
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, weights=None, **kwargs):
        """
        Complex-valued FIR envelope detector based on analytic signal reconstruction
        :param band: freq. range to apply band-pass filtering
        :param fs: sampling frequency
        :param delay: delay of ideal filter in samples
        :param n_taps: length of FIR
        :param n_fft: length of freq. grid to estimate ideal freq. response
        :weights: least squares weights. If None match WHilbertFilter
        """
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * fs < band[0]) | (w / n_fft * fs > band[1])] = 0
        F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
        if weights is None:
            self.b = F.T.conj().dot(H)/n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b)-1)

    def apply(self, chunk: np.ndarray):
        y, self.zi = sg.lfilter(self.b, self.a, chunk, zi=self.zi)
        return y
