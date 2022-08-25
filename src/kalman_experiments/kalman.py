from cmath import exp
from typing import NamedTuple

import numpy as np

from .complex import complex2mat, vec2complex
from .numpy_types import Cov, Mat, Vec, Vec1D


class ColoredNoiseKF:
    """
    'Alternative approach' implementation for KF with colored noise from [1]

    Parameters
    ----------
    n_x : int
        Number of channels in the state vector
    n_z : int
        Number of channels in measurements
    Phi : np.ndarray of shape(n_x, n_x)
        State transfer matrix
    Q : np.ndarray of shape(n_x, n_x)
        Process noise covariance matrix (see eq.(1) in [1])
    H : np.ndarray of shape(n_z, n_x)
        Matrix of the measurements model (see eq.(2) in [1]); maps state to
        measurements
    Psi : np.ndarray of shape(n_z, n_z)
        Measurement noise transfer matrix (see eq. (3) in [1])
    R : np.ndarray of shape(n_z, n_z)
        Driving noise covariance matrix for the noise AR model (cov for e_{k-1}
        in eq. (3) in [1])

    References
    ----------
    .. [1] Chang, G. "On kalman filter for linear system with colored
    measurement noise". J Geod 88, 1163–1170, 2014
    https://doi.org/10.1007/s00190-014-0751-7

    """

    def __init__(self, n_x: int, n_z: int, Phi: Mat, Q: Cov, H: Mat, Psi: Mat, R: Cov):
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.Psi = Psi
        self.R = R

        self.x = np.zeros((n_x, 1))  # posterior state (after update)
        self.P = np.zeros((n_x, n_x))  # posterior state covariance (after update)

        self.y_prev = np.zeros((n_z, 1))

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


class Gaussian(NamedTuple):
    mu: Vec
    Sigma: Cov


class Colored1DMatsudaKF:
    """
    Single oscillation - single measurement Kalman filter with colored noise

    Using Matsuda's model for oscillation prediction, see [1], and AR(1) to
    make account for 1/f^a measurement noise, see [2]. Wraps ColoredNoiseKF to
    avoid trouble with properly arranging matrix and vector shapes.

    Parameters
    ----------
    A : float
        A in Matsuda's step equation: x_next = A * exp(2 * pi * i * f / sr) * x + n
    f : float
        Oscillation frequency; f in Matsuda's step equation:
        x_next = A * exp(2 * pi * i * f / sr) * x + n
    sr : float
        Sampling rate
    q_sigma : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : float
        Coefficient of the AR(1) process modelling 1/f^a colored noise;
        see eq. (3) in [2]; -0.5 corresponds to 1/f noise, 0 -- to white noise
    r_sigma : float
        Driving white noise covariance matrix for the noise AR model
        (cov for e_{k-1} in eq. (3) in [2])

    References
    ----------
    .. [1] Matsuda, Takeru, and Fumiyasu Komaki. “Time Series Decomposition
    into Oscillation Components and Phase Estimation.” Neural Computation 29,
    no. 2 (February 2017): 332–67. https://doi.org/10.1162/NECO_a_00916.

    .. [2] Chang, G. "On kalman filter for linear system with colored
    measurement noise". J Geod 88, 1163–1170, 2014
    https://doi.org/10.1007/s00190-014-0751-7

    """

    def __init__(self, A: float, f: float, sr: float, q_sigma: float, psi: float, r_sigma: float):
        Phi = complex2mat(A * exp(2 * np.pi * f / sr * 1j))
        Q = np.eye(2) * q_sigma**2
        H = np.array([[1, 0]])
        Psi = np.array([[psi]])
        R = np.array([[r_sigma**2]])
        self.KF = ColoredNoiseKF(n_x=2, n_z=1, Phi=Phi, Q=Q, H=H, Psi=Psi, R=R)

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


class KFAdapter:
    def __init__(self, kf: Colored1DMatsudaKF):
        self.kf = kf

    def apply(self, signal: Vec1D, delay: int) -> Vec1D:
        if delay > 0:
            raise NotImplementedError("Kalman smoothing is not implemented")
        res = []
        for y in signal:
            state = self.kf.step(y)
            for _ in range(abs(delay)):
                state = self.kf.predict(state)
            res.append(vec2complex(state.mu))
        return np.array(res)
