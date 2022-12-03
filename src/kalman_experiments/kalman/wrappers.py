"""
Kalman filter implementations

References
----------
.. [1] Matsuda, Takeru, and Fumiyasu Komaki. “Time Series Decomposition into
Oscillation Components and Phase Estimation.” Neural Computation 29, no. 2
(February 2017): 332–67. https://doi.org/10.1162/NECO_a_00916.

.. [2] Chang, G. "On kalman filter for linear system with colored measurement
noise". J Geod 88, 1163–1170, 2014 https://doi.org/10.1007/s00190-014-0751-7

.. [3] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to Kalman
Filtering with Time-Correlated Measurement Errors.” IEEE Transactions on
Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
https://doi.org/10.1109/TAES.2012.6178086.

.. [4] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings of
the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.

"""
from abc import ABC
from cmath import exp
from typing import Any, NamedTuple

import numpy as np

from ..complex import complex2mat, vec2complex
from ..models import MatsudaParams
from ..numpy_types import Cov, Vec, Vec1D
from .core import DifferenceColoredKF, PerturbedPKF


class Gaussian(NamedTuple):
    mu: Vec
    Sigma: Cov


class OneDimKF(ABC):
    """Single measurement Kalman filter abstraction"""

    KF: Any

    def predict(self, X: Gaussian) -> Gaussian:
        return Gaussian(*self.KF.predict(X.mu, X.Sigma))

    def update(self, y: float, X_: Gaussian) -> Gaussian:
        y_arr = np.array([[y]])
        return Gaussian(*self.KF.update(y=y_arr, x_=X_.mu, P_=X_.Sigma))

    def update_no_meas(self, X_: Gaussian) -> Gaussian:
        """Update step when the measurement is missing"""
        return Gaussian(*self.KF.update_no_meas(x_=X_.mu, P_=X_.Sigma))

    def step(self, y: float | None) -> Gaussian:
        """Predict and update in one step"""
        X_ = self.predict(Gaussian(self.KF.x, self.KF.P))
        return self.update_no_meas(X_) if y is None else self.update(y, X_)


class Difference1DMatsudaKF(OneDimKF):
    """
    Single oscillation - single measurement Kalman filter with AR(1) colored noise

    Using Matsuda's model for oscillation prediction, see [1], and a difference
    scheme to incorporate AR(1) 1/f^a measurement noise, see [2]. Wraps
    DifferenceColoredKF to avoid trouble with properly arranging matrix and
    vector shapes.

    Parameters
    ----------
    A : float
        A in Matsuda's step equation: x_next = A * exp(2 * pi * i * f / sr) * x + n
    f : float
        Oscillation frequency; f in Matsuda's step equation:
        x_next = A * exp(2 * pi * i * f / sr) * x + n
    sr : float
        Sampling rate
    q_s : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : float
        Coefficient of the AR(1) process modelling 1/f^a colored noise;
        see eq. (3) in [2]; 0.5 corresponds to 1/f noise, 0 -- to white noise,
        1 -- to Brownian motion, see [4]. In between values are also allowed.
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])

    See also
    --------
    gen_ar_noise_coefficients : generate psi

    """

    def __init__(self, A: float, f: float, sr: float, q_s: float, psi: float, r_s: float):
        Phi = complex2mat(A * exp(2 * np.pi * f / sr * 1j))
        Q = np.eye(2) * q_s**2
        H = np.array([[1, 0]])
        Psi = np.array([[psi]])
        R = np.array([[r_s**2]])
        self.KF = DifferenceColoredKF(Phi=Phi, Q=Q, H=H, Psi=Psi, R=R)


class PerturbedP1DMatsudaKF(OneDimKF):
    """
    Single oscillation - single measurement Kalman filter with AR(n_ar) colored noise

    Using Matsuda's model for oscillation prediction, see [1], and AR(n) to
    make account for 1/f^a measurement noise. Previous states for
    AR(n_ar) are included via state-space augmentation with the Perturbed P
    stabilization technique, see [3]. Wraps PerturbedPKF to avoid trouble with
    properly arranging matrix and vector shapes.

    Parameters
    ----------
    M : MatsudaParams
    q_s : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : np.ndarray of shape(n_ar,)
        Coefficients of the AR(n_ar) process modelling 1/f^a colored noise;
        used to set up Psi as in eq. (3) in [2];
        coefficients correspond to $-a_i$ in eq. (115) in [4]
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3]

    See also
    --------
    gen_ar_noise_coefficients : generate psi

    """

    def __init__(
        self,
        M: MatsudaParams,
        q_s: float,
        psi: np.ndarray,
        r_s: float,
        lambda_: float = 1e-6,
    ):
        ns = len(psi)  # number of noise states

        Phi_blocks = [
            [complex2mat(M.A * exp(2 * np.pi * M.freq / M.sr * 1j)), np.zeros([2, ns])],
        ]
        if ns:
            Phi_blocks.append([np.zeros([1, 2]), psi[np.newaxis, :]])
            Phi_blocks.append([np.zeros([ns - 1, 2]), np.eye(ns - 1), np.zeros([ns - 1, 1])])
        Phi = np.block(Phi_blocks)  # pyright: ignore
        Q_blocks = [[np.eye(2) * q_s**2, np.zeros([2, ns])]]
        if ns:
            Q_noise = np.zeros([ns, ns])
            Q_noise[0, 0] = r_s**2
            Q_blocks.append([np.zeros([ns, 2]), Q_noise])
        Q = np.block(Q_blocks)  # pyright: ignore

        H_noise = [1] + [0] * (ns - 1) if ns else []
        H = np.array([[1, 0] + H_noise])
        if ns:
            R = np.array([[0]])
        else:
            R = np.array([[r_s**2]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=lambda_)
        self.M = M
        self.psi = psi
        self.lambda_ = lambda_
        self.q_s = q_s
        self.r_s = r_s

    def __repr__(self) -> str:
        return (
            f"PerturbedP1DMatsudaKF(M={self.M}, q_s={self.q_s:.2f},"
            f" psi={self.psi}, r_s={self.r_s:.2f}, lambda_={self.lambda_})"
        )


class PerturbedP1DMatsudaSmoother(OneDimKF):
    def __init__(
        self,
        M: MatsudaParams,
        q_s: float,
        psi: np.ndarray,
        r_s: float,
        lag: int = 0,
        lambda_: float = 1e-6,
    ):
        lag = 0 if lag < 0 else lag
        n_aug_x = lag * 2
        ns = len(psi)  # number of noise states

        Phi = np.block(
            [  # pyright: ignore
                [complex2mat(M.Phi), np.zeros([2, ns + n_aug_x])],
                [np.eye(n_aug_x), np.zeros([n_aug_x, 2 + ns])],
                [np.zeros([1, n_aug_x + 2]), psi[np.newaxis, :]],
                [np.zeros([ns - 1, n_aug_x + 2]), np.eye(ns - 1), np.zeros([ns - 1, 1])],
            ]
        )
        Q_noise = np.zeros([ns, ns])
        Q_noise[0, 0] = r_s**2
        Q = np.block(
            [  # pyright: ignore
                [np.eye(2) * q_s**2, np.zeros([2, ns + n_aug_x])],
                [np.zeros([n_aug_x, n_aug_x + ns + 2])],
                [np.zeros([ns, n_aug_x + 2]), Q_noise],
            ]
        )

        H = np.array([[1, 0] + [0] * n_aug_x + [1] + [0] * (ns - 1)])
        R = np.array([[0]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=lambda_)
        self.lag = lag


def apply_kf(kf: OneDimKF, signal: Vec1D, delay: int) -> Vec1D:
    """Convenience function to filter all signal samples at once with KF"""
    res = []
    # AR_ORDER = 2
    if delay > 0:
        assert hasattr(kf, "lag"), "Smoothing is not implemented for this KF"
        assert kf.lag <= delay  # pyright: ignore
        for y in signal:
            state = kf.step(y)
            res.append(vec2complex(state.mu[delay * 2 : (delay + 1) * 2]))
    else:
        k = 0
        for y in signal:
            state = kf.step(y)
            # envs = np.abs([vec2complex(state.mu[i * 2:(i + 1) * 2]) for i in range(5)])
            # rho, _ = yule_walker(envs, order=AR_ORDER)
            #     print(f"{rho=}, {envs=}")
            # envs_ar = list(envs[:AR_ORDER])
            # new_env = envs[0]
            for _ in range(abs(delay)):
                state = kf.predict(state)
                # new_env = rho.dot(envs_ar)
                # new_env += de
                # de += de2
                # envs_ar = [new_env] + envs_ar[:-1]
                # if not k % 500:
                #     print(f"{envs_ar=}")
            k += 1
            pred = vec2complex(state.mu[:2])
            # pred /= np.abs(pred)
            # pred *= new_env
            res.append(pred)
    return np.array(res)
