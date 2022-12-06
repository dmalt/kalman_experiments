"""Vector general-case kalman filter implementations"""
import numpy as np

from ..numpy_types import Cov, Mat, Vec


class DifferenceKF:
    """
    'Alternative approach' implementation for KF with colored noise from [1]

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

        self.x = np.zeros((n_states, 1))  # posterior state (after update)
        # self.P = np.zeros((n_states, n_states))  # posterior state covariance (after update)
        self.P = np.eye(n_states)  # posterior cov (after update)

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

    References
    ----------
    .. [1] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to Kalman
    Filtering with Time-Correlated Measurement Errors.” IEEE Transactions on
    Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
    https://doi.org/10.1109/TAES.2012.6178086.

    """

    def __init__(
        self, Phi: Mat, Q: Cov, H: Mat, R: Cov, x_0: Vec | None = None, P_0: Cov | None = None
    ):
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.R = R

        n_states = Phi.shape[0]
        self.x = np.zeros((n_states, 1)) if x_0 is None else x_0  # posterior state (after update)
        self.P = np.eye(n_states) if P_0 is None else P_0  # posterior cov (after update)

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
    Perturbed P implementation from [1] for KF with augmented state space

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

    References
    ----------
    .. [1] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to Kalman
    Filtering with Time-Correlated Measurement Errors.” IEEE Transactions on
    Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
    https://doi.org/10.1109/TAES.2012.6178086.

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, R: Cov, lambda_: float = 1e-6):
        super().__init__(Phi, Q, H, R)
        self.lambda_ = lambda_

    def update(self, y: Vec, x_: Vec, P_: Cov) -> tuple[Vec, Cov]:
        super().update(y, x_, P_)
        self.P += np.eye(len(self.P)) * self.lambda_
        return self.x, self.P
