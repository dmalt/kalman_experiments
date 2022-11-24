"""General kalman filter implementations"""
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from typing import Sequence

import numpy as np

from ..numpy_types import Cov, Mat, Vec


@dataclass(frozen=True)
class Gaussian:
    x: Vec
    P: Cov

    def __post_init__(self):
        n = len(self.x)
        assert self.P.shape == (n, n), f"Incompatible sizes: {self.x.shape=}, {self.P.shape=}"

    def __len__(self) -> int:
        return len(self.x)


class BaseKF(ABC):
    """
    Abstract base class for Kalman filter

    Attributes
    ----------
    state : Gaussian
        State vector and its covariance matrix
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model; maps state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model

    Notes
    -----
    state = posterior state (after the update step)
    state_ = prior state (after the predict step)

    """

    state: Gaussian
    Phi: Mat
    Q_s: Cov
    H: Mat
    R_s: Cov

    @abstractmethod
    def predict(self, state: Gaussian) -> Gaussian:
        """Predict next state according to the model"""

    @abstractmethod
    def update(self, y: Vec, state_: Gaussian) -> Gaussian:
        """Update prediction based on the measurement"""

    @abstractmethod
    def update_no_meas(self, state_: Gaussian) -> Gaussian:
        """Update step when the measurement is missing"""

    def step(self, y: Vec | None) -> Gaussian:
        """Predict-update combo in one step, possibly with missing data"""
        state_ = self.predict(self.state)
        assert not np.any(np.isnan(state_.x)) and not np.any(np.isnan(state_.P))
        state = self.update(y, state_) if y is not None else self.update_no_meas(state_)
        assert not np.any(np.isnan(state.x)) and not np.any(np.isnan(state.P))
        return state

    def apply(self, all_y: Sequence[Vec]) -> list[Gaussian]:
        """Apply KF to measured timeseries"""
        res = [self.state]
        for t, y in enumerate(all_y):
            res.append(self.step(y))
        return res
        # return [self.state] + [self.step(y) for y in all_y]


class SimpleKF(BaseKF):
    """
    Standard Kalman filter implementation

    Implementation follows eq. (2, 3) from [1]

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
    init_state : Gaussian, optional, default=None
        Initial state vector and its covariance

    References
    ----------
    .. [1] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to
    Kalman Filtering with Time-Correlated Measurement Errors.” IEEE
    Transactions on Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
    https://doi.org/10.1109/TAES.2012.6178086.

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, R: Cov, init_state: Gaussian | None = None):
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.R = R

        n_states = Phi.shape[0]
        if init_state is None:
            # self.state = Gaussian(np.zeros([n_states, 1]), np.eye(n_states) * 1e-3)
            self.state = Gaussian(np.random.randn(n_states, 1), np.eye(n_states) * 1e-3)
        else:
            self.state = init_state

    def predict(self, state: Gaussian) -> Gaussian:
        x_ = self.Phi @ state.x
        P_ = self.Phi @ state.P @ self.Phi.T + self.Q
        return Gaussian(x_, P_)

    def update(self, y: Vec, state_: Gaussian) -> Gaussian:
        Sigma = self.H @ state_.P @ self.H.T + self.R
        Pxn = state_.P @ self.H.T

        K = np.linalg.solve(Sigma, Pxn.T).T
        n = y - self.H @ state_.x
        self.state = Gaussian(state_.x + K @ n, state_.P - K @ Sigma @ K.T)
        return self.state

    def update_no_meas(self, state_: Gaussian) -> Gaussian:
        """Update step when the measurement is missing"""
        self.state = state_
        return state_

    # def step(self, y: Vec | None) -> Gaussian:
    #     state_ = self.predict(self.state)
    #     return self.update(y, state_) if y is not None else self.update_no_meas(state_)


class DifferenceKF(BaseKF):
    """
    'Alternative approach' implementation for KF with colored noise from [2]

    Parameters
    ----------
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix (see eq.(1) in [1])
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model (see eq.(2) in [1]); maps state to
        measurements
    Psi : np.ndarray of shape(n_meas, n_meas)
        Measurement noise transfer matrix (see eq. (3) in [1])
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model (cov for e_{k-1}
        in eq. (3) in [1])

    References
    ----------
    .. [1] Chang, G. "On kalman filter for linear system with colored measurement
    noise". J Geod 88, 1163–1170, 2014 https://doi.org/10.1007/s00190-014-0751-7
    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, Psi: Mat, R: Cov):
        n_states = Phi.shape[0]
        n_meas = H.shape[0]

        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.Psi = Psi
        self.R = R

        self.state = Gaussian(np.zeros((n_states, 1)), np.zeros((n_states, n_states)))
        self.y_prev = np.zeros((n_meas, 1))

    def predict(self, state: Gaussian) -> Gaussian:
        x_ = self.Phi @ state.x  # eq. (26) from [1]
        P_ = self.Phi @ state.P @ self.Phi.T + self.Q  # eq. (27) from [1]
        return Gaussian(x_, P_)

    def update(self, y: Vec, state_: Gaussian) -> Gaussian:
        A = self.Psi @ self.H
        B = self.H @ self.Phi
        H, R = self.H, self.R
        x, P = astuple(self.state)
        x_, P_ = astuple(state_)

        z = y - self.Psi @ self.y_prev  # eq. (35) from [1]
        n = z - self.H @ x_ + A @ x  # eq. (37) from [1]
        Sigma = H @ P_ @ H.T + A @ P @ A.T + R - B @ P @ A.T - A @ P @ B.T  # eq. (38) from [1]
        Pxn = P_ @ self.H.T - self.Phi @ P @ A.T  # eq. (39) from [1]

        K = Pxn / Sigma  # eq. (40) from [1]
        x = x_ + K * n  # eq. (41) from [1]
        P = P_ - K * Sigma @ K.T  # eq. (42) from [1]
        self.state = Gaussian(x, P)
        self.y_prev = y
        return self.state

    def update_no_meas(self, state_: Gaussian) -> Gaussian:
        """Update step when the measurement is missing"""
        self.state = state_
        self.y_prev = self.H @ state_.x
        return state_

    def step(self, y: Vec | None) -> Gaussian:
        state_ = self.predict(self.state)
        return self.update(y, state_) if y is not None else self.update_no_meas(state_)


class PerturbedPKF(SimpleKF):
    """
    Perturbed P implementation from [1] for KF with augmented state space

    Parameters
    ----------
    Phi : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented state transfer matrix (see eq. (9) in [1])
    Q : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented process noise covariance matrix (see eq.(9) in [1])
    H : np.ndarray of shape(n_meas, n_aug_states)
        Augmented matrix of the measurements model (see eq.(9) in [1]); maps
        augmented state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Measurements covariance matrix, usually of zeroes, see notes
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [1].

    Notes
    -----
    R is added for possible regularization and normally must be a zero matrix,
    since the measurement errors are incorporated into the augmented state
    vector

    References
    ----------
    .. [1] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to
    Kalman Filtering with Time-Correlated Measurement Errors.” IEEE
    Transactions on Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
    https://doi.org/10.1109/TAES.2012.6178086.

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, R: Cov, lambda_: float = 1e-6):
        super().__init__(Phi, Q, H, R)
        self.lambda_ = lambda_

    def update(self, y: Vec, state_: Gaussian) -> Gaussian:
        super().update(y, state_)
        self.state = Gaussian(self.state.x, self.state.P + np.eye(len(self.state)) * self.lambda_)
        return self.state
