"""

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

from .complex import complex2mat, vec2complex
from .numpy_types import Cov, Mat, Vec, Vec1D


class DifferenceColoredKF:
    """
    'Alternative approach' implementation for KF with colored noise from [2]

    Parameters
    ----------
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix (see eq.(1) in [2])
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model (see eq.(2) in [2]); maps state to
        measurements
    Psi : np.ndarray of shape(n_meas, n_meas)
        Measurement noise transfer matrix (see eq. (3) in [2])
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model (cov for e_{k-1}
        in eq. (3) in [2])

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, Psi: Mat, R: Cov):
        n_states = Phi.shape[0]
        n_meas = H.shape[0]

        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.Psi = Psi
        self.R = R

        self.x = np.zeros((n_states, 1))  # posterior state (after update)
        self.P = np.zeros((n_states, n_states))  # posterior state covariance (after update)

        self.y_prev = np.zeros((n_meas, 1))

    def predict(self, x: Vec, P: Cov) -> tuple[Vec, Cov]:
        x_ = self.Phi @ x  # eq. (26) from [1]
        P_ = self.Phi @ P @ self.Phi.T + self.Q  # eq. (27) from [1]
        return x_, P_

    def update(self, y: Vec, x_: Vec, P_: Cov) -> tuple[Vec, Cov]:
        A = self.Psi @ self.H
        B = self.H @ self.Phi
        P, H, R = self.P, self.H, self.R

        z = y - self.Psi @ self.y_prev  # eq. (35) from [1]
        n = z - self.H @ x_ + A @ self.x  # eq. (37) from [1]
        Sigma = H @ P_ @ H.T + A @ P @ A.T + R - B @ P @ A.T - A @ P @ B.T  # eq. (38) from [1]
        Pxn = P_ @ self.H.T - self.Phi @ P @ A.T  # eq. (39) from [1]

        K = Pxn / Sigma  # eq. (40) from [1]
        self.x = x_ + K * n  # eq. (41) from [1]
        self.P = P_ - K * Sigma @ K.T  # eq. (42) from [1]
        self.y_prev = y
        return self.x, self.P

    def update_no_meas(self, x_: Vec, P_: Cov):
        """Update step when the measurement is missing"""
        self.x = x_
        self.P = P_
        self.y_prev = self.H @ x_
        return x_, P_

    def step(self, y: Vec | None) -> tuple[Vec, Cov]:
        x_, P_ = self.predict(self.x, self.P)
        return self.update(y, x_, P_) if y is not None else self.update_no_meas(x_, P_)


class SimpleKF:
    """
    Standard Kalman filter implementation

    Implementation follows eq. (2, 3) from [3]

    Parameters
    ----------
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model; maps state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, R: Cov):
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.R = R

        n_states = Phi.shape[0]
        self.x = np.zeros((n_states, 1))  # posterior state (after update)
        self.P = np.zeros((n_states, n_states))  # posterior state covariance (after update)

    def predict(self, x: Vec, P: Cov) -> tuple[Vec, Cov]:
        x_ = self.Phi @ x
        P_ = self.Phi @ P @ self.Phi.T + self.Q
        return x_, P_

    def update(self, y: Vec, x_: Vec, P_: Cov) -> tuple[Vec, Cov]:
        Sigma = self.H @ P_ @ self.H.T + self.R
        Pxn = P_ @ self.H.T

        K = Pxn / Sigma
        n = y - self.H @ x_
        self.x = x_ + K @ n
        self.P = P_ - K @ Sigma @ K.T
        return self.x, self.P

    def update_no_meas(self, x_: Vec, P_: Cov):
        """Update step when the measurement is missing"""
        self.x = x_
        self.P = P_
        return x_, P_

    def step(self, y: Vec | None) -> tuple[Vec, Cov]:
        x_, P_ = self.predict(self.x, self.P)
        return self.update(y, x_, P_) if y is not None else self.update_no_meas(x_, P_)


class PerturbedPKF(SimpleKF):
    """
    Perturbed P implementation from [3] for KF with augmented state space

    Parameters
    ----------
    Phi : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented state transfer matrix (see eq. (9) in [3])
    Q : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented process noise covariance matrix (see eq.(9) in [3])
    H : np.ndarray of shape(n_meas, n_aug_states)
        Augmented matrix of the measurements model (see eq.(9) in [3]); maps
        augmented state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Measurements covariance matrix, usually of zeroes, see notes
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3].

    Notes
    -----
    R is added for possible regularization and normally must be a zero matrix,
    since the measurement errors are incorporated into the augmented state
    vector

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, R: Cov, lambda_: float = 1e-6):
        super().__init__(Phi, Q, H, R)
        self.lambda_ = lambda_

    def update(self, y: Vec, x_: Vec, P_: Cov) -> tuple[Vec, Cov]:
        super().update(y, x_, P_)
        self.P += np.eye(len(self.P)) * self.lambda_
        return self.x, self.P


class Gaussian(NamedTuple):
    mu: Vec
    Sigma: Cov


class OneDimKF(ABC):
    """Single oscillation - single measurement Kalman filter abstraction"""
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
    psi : np.ndarray of shape(n_ar,)
        Coefficients of the AR(n_ar) process modelling 1/f^a colored noise;
        used to set up Psi as in eq. (3) in [2];
        coefficients correspond to $-a_i$ in eq. (115) in [4]
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3]

    """

    def __init__(
        self,
        A: float,
        f: float,
        sr: float,
        q_s: float,
        psi: np.ndarray,
        r_s: float,
        lambda_: float = 1e-6,
    ):
        ns = len(psi)  # number of noise states

        Phi = np.block(
            [  # pyright: ignore
                [complex2mat(A * exp(2 * np.pi * f / sr * 1j)), np.zeros([2, ns])],
                [np.zeros([1, 2]), psi[np.newaxis, :]],
                [np.zeros([ns - 1, 2]), np.eye(ns - 1), np.zeros([ns - 1, 1])],
            ]
        )
        Q_noise = np.zeros([ns, ns])
        Q_noise[0, 0] = r_s**2
        Q = np.block(
            [  # pyright: ignore
                [np.eye(2) * q_s**2, np.zeros([2, ns])],
                [np.zeros([ns, 2]), Q_noise],
            ]
        )

        H = np.array([[1, 0, 1] + [0] * (ns - 1)])
        R = np.array([[0]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=lambda_)


def apply_kf(kf: OneDimKF, signal: Vec1D, delay: int) -> Vec1D:
    """Convenience function to filter all signal samples at once with KF"""
    if delay > 0:
        raise NotImplementedError("Kalman smoothing is not implemented")
    res = []
    for y in signal:
        state = kf.step(y)
        for _ in range(abs(delay)):
            state = kf.predict(state)
        res.append(vec2complex(state.mu))
    return np.array(res)
