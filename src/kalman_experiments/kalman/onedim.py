"""
Kalman filter implementations

Notes
-----
Soss = single oscillation - single sensor

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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..complex import complex2mat, vec2complex
from ..models import MatsudaParams
from ..numpy_types import (
    Cov,
    Mat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Vec1D,
    check_nonnegative_int,
    check_positive_int,
    is_nonnegative_int,
)
from .core import BaseKF, DifferenceKF, Gaussian, PerturbedPKF


class SingleSensorKf(ABC):
    """Single measurement Kalman filter abstraction"""

    KF: BaseKF

    def step(self, y: float | None, osc_ind: NonNegativeInt = check_nonnegative_int(0)) -> complex:
        y_arr = y if y is None else np.array([[y]])
        state = self.KF.step(y_arr)
        return self._get_oscillation(state, osc_ind)

    def apply(
        self, all_y: Sequence[float], osc_ind: NonNegativeInt = check_nonnegative_int(0)
    ) -> list[complex]:
        return [self.step(y, osc_ind) for y in all_y]

    def predict(
        self,
        n_steps: PositiveInt = check_positive_int(1),
        osc_ind: NonNegativeInt = check_nonnegative_int(0),
    ) -> complex:
        state = self.KF.state
        for _ in range(n_steps):
            state = self.KF.predict(state)
        return self._get_oscillation(state, osc_ind)

    @property
    @abstractmethod
    def n_oscillations(self) -> PositiveInt:
        """
        Number of oscillations in a model

        I.e. number of oscillatory components in the MK model or a number
        of augmented oscillatory state pairs for smoother

        """

    def _get_oscillation(self, state: Gaussian, osc_ind: NonNegativeInt) -> complex:
        assert osc_ind < self.n_oscillations
        assert self.n_oscillations <= len(state) // 2
        return vec2complex(state.x[osc_ind * 2 : (osc_ind + 1) * 2])


@dataclass()
class SossDifferenceKf(SingleSensorKf):
    """
    Single oscillation - single measurement Kalman filter with AR(1) colored noise

    Using Matsuda's model for oscillation prediction, see [1], and a difference
    scheme to incorporate AR(1) 1/f^a measurement noise, see [2]. Wraps
    DifferenceKF to avoid trouble with properly arranging matrix and
    vector shapes.

    Parameters
    ----------
    mp : MatsudaParams
        Matsuda model parameters
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

    """

    mp: MatsudaParams
    q_s: NonNegativeFloat
    psi: float
    r_s: NonNegativeFloat
    KF: DifferenceKF = field(init=False)

    def __post_init__(self) -> None:
        Phi = complex2mat(self.mp.Phi)
        Q = np.eye(2) * self.q_s**2
        H = np.array([[1, 0]])
        Psi = np.array([[self.psi]])
        R = np.array([[self.r_s**2]])
        self.KF = DifferenceKF(Phi=Phi, Q=Q, H=H, Psi=Psi, R=R)

    @property
    def n_oscillations(self) -> PositiveInt:
        return check_positive_int(1)


@dataclass
class SossAugSmoother(SingleSensorKf):
    """
    Single oscillation - single measurement fixed lag smoother for colored noise

    Uses Matsuda's model for oscillation prediction, see [1], and AR(n) to make
    account for 1/f^a measurement noise. Previous autoregressive noise and
    smoothing MK-model states states are included via state-space augmentation
    with the Perturbed P stabilization technique, see [3]. Wraps PerturbedPKF
    to avoid trouble with properly arranging matrix and vector shapes.

    Parameters
    ----------
    mp : MatsudaParams
    q_s : float, >= 0
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : np.ndarray of shape(n_ar,)
        Coefficients of the AR(n_ar) process modelling 1/f^a colored noise;
        used to set up Psi as in eq. (3) in [2];
        coefficients correspond to $-a_i$ in eq. (115) in [4]
    r_s : float >= 0
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])
    lag : int, >= 0, default=0
        Smoothing lag. If lag=0, operates as Kalman filter
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3]

    """

    mp: MatsudaParams
    q_s: NonNegativeFloat
    psi: np.ndarray
    r_s: NonNegativeFloat
    lag: NonNegativeInt = check_nonnegative_int(0)
    lambda_: float = 1e-6
    KF: PerturbedPKF = field(init=False)

    def __post_init__(self) -> None:
        n_aug_x = check_nonnegative_int(self.lag * 2)
        ns = check_nonnegative_int(len(self.psi))  # number of noise states
        Phi = self._assemble_Phi(ns, n_aug_x)
        Q = self._assemble_Q(ns, n_aug_x)
        H = self._assemble_H(ns, n_aug_x)
        R = np.array([[0]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=self.lambda_)

    @property
    def n_oscillations(self) -> PositiveInt:
        return check_positive_int(self.lag + 1)

    def _assemble_Phi(self, ns: NonNegativeInt, n_aug_x: NonNegativeInt) -> Mat:
        return np.block(
            [  # pyright: ignore
                [complex2mat(self.mp.Phi), np.zeros([2, ns + n_aug_x])],
                [np.eye(n_aug_x), np.zeros([n_aug_x, 2 + ns])],
                [np.zeros([1, n_aug_x + 2]), self.psi[np.newaxis, :]],
                [np.zeros([ns - 1, n_aug_x + 2]), np.eye(ns - 1), np.zeros([ns - 1, 1])],
            ]
        )

    def _assemble_Q(self, ns: NonNegativeInt, n_aug_x: NonNegativeInt) -> Cov:
        Q_noise = np.zeros([ns, ns])
        Q_noise[0, 0] = self.r_s**2
        return np.block(
            [  # pyright: ignore
                [np.eye(2) * self.q_s**2, np.zeros([2, ns + n_aug_x])],
                [np.zeros([n_aug_x, n_aug_x + ns + 2])],
                [np.zeros([ns, n_aug_x + 2]), Q_noise],
            ]
        )

    def _assemble_H(self, ns: NonNegativeInt, n_aug_x: NonNegativeInt) -> Mat:
        return np.array([[1, 0] + [0] * n_aug_x + [1] + [0] * (ns - 1)])


def apply_kf(kf: SingleSensorKf, signal: Sequence[float], delay: int) -> Vec1D:
    """Convenience function to filter all signal samples at once with KF"""
    if is_nonnegative_int(delay) and isinstance(kf, SossAugSmoother):
        return np.array(kf.apply(signal, osc_ind=delay))
    elif not is_nonnegative_int(delay):
        res = []
        for y in signal:
            kf.step(y)
            res.append(kf.predict(n_steps=check_positive_int(abs(delay))))
        return np.array(res)
    else:
        raise TypeError("Unsupported KF type")
