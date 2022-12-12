"""
Vector general-case kalman filter implementations


Examples
--------
Get smoothed data
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from kalman_experiments.kalman.wrappers import PerturbedP1DMatsudaKF
>>> from kalman_experiments.models import MatsudaParams, SingleRhythmModel, collect
>>> from kalman_experiments.model_selection import fit_kf_parameters

Setup oscillatioins model and generate oscillatory signal
>>> mp = MatsudaParams(A=0.99, freq=10, sr=1000)
>>> gt_states = collect(SingleRhythmModel(mp, sigma=1), n_samp=1000)
>>> meas = np.real(gt_states) + np.random.randn(len(gt_states))
>>> kf = PerturbedP1DMatsudaKF(mp, q_s=1, psi=np.zeros(0), r_s=10, lambda_=0).KF
>>> y = normalize_measurement_dimensions(meas)
>>> x, P = apply_kf(kf, y)
>>> x_n, P_n, J = apply_kalman_interval_smoother(kf, x, P)
>>> res = plt.plot([xx[0] for xx in x], label="fp", linewidth=4)
>>> res = plt.plot([xxn[0] for xxn in x_n], label="smooth", linewidth=4)
>>> res = plt.plot(np.real(gt_states), label="gt", linewidth=2)
>>> l = plt.legend()
>>> plt.show()

"""
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

        K = np.linalg.solve(Sigma, Pxn.T).T
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


def apply_kf(KF: SimpleKF, y: list[Vec]) -> tuple[list[Vec], list[Cov]]:
    n = len(y) - 1
    x: list[Vec] = [None] * (n + 1)  # type: ignore  # x^t_t
    P: list[Cov] = [None] * (n + 1)  # type: ignore  # P^t_t
    x[0], P[0] = KF.x, KF.P
    for t in range(1, n + 1):
        x[t], P[t] = KF.step(y[t])
    return x, P


def apply_kalman_interval_smoother(
    KF: SimpleKF, x: list[Vec], P: list[Cov]
) -> tuple[list[Vec], list[Cov], list[Mat]]:
    n = len(x) - 1
    x_n: list[Vec] = [None] * (n + 1)  # type: ignore  # x^n_t
    P_n: list[Cov] = [None] * (n + 1)  # type: ignore  # P^n_t
    x_n[n], P_n[n] = x[n], P[n]
    J: list[Mat] = [None] * (n + 1)  # type: ignore
    for t in range(n, 0, -1):
        x_n[t - 1], P_n[t - 1], J[t - 1] = smoother_step(KF, x[t - 1], P[t - 1], x_n[t], P_n[t])

    return x_n, P_n, J


def smoother_step(KF: SimpleKF, x: Vec, P: Cov, x_n: Vec, P_n: Cov) -> tuple[Vec, Cov, Mat]:
    """
    Make one Kalman Smoother step

    Parameters
    ----------
    x : Vec
        State estimate after KF update step after the forward pass, i.e.
        x^{t-1}_{t-1} in eq (6.47) in [1]
    P : Cov
        State covariance after KF update step after the forward pass, i.e.
        P^{t-1}_{t-1} in eq. (6.48) in [1]
    x_n : Vec
        Smoothed state estimate for the time instaint following the one being
        currently processed, i.e. x^{n}_{t} in eq. (6.47) in [1]
    P_n : Cov
        Smoothed state covariance for the time instant following the one being
        currently processed, i.e. P^{n}_{t} in eq. (6.47) in [1]

    Returns
    -------
    x_n : Vec
        Smoothed state estimate for one timestep back, i.e. x^{n}_{t-1} in eq.
        (6.47) in [1]
    P_n : Cov
        Smoothed state covariance for one timestep back, i.e. P^{n}_{t-1} in eq. (6.48) in [1]
    J : Mat
        J_{t-1} in eq. (6.49) in [1]

    Notes
    -----
    Code here follows slightly different notation than in em_step(); e.g. here
    x_n is a state vector for a single time instant compared to an array of
    state vectors in em_step().

    References
    ----------
    [1] .. Shumway, Robert H., and David S. Stoffer. 2011. Time Series Analysis
    and Its Applications. Springer Texts in Statistics. New York, NY: Springer
    New York. https://doi.org/10.1007/978-1-4419-7865-3.

    """
    x_, P_ = KF.predict(x, P)

    J = np.linalg.solve(P_, KF.Phi @ P).T  # P * Phi^T * P_^{-1}; solve is better than inv

    x_n = x + J @ (x_n - x_)
    P_n = P + J @ (P_n - P_) @ J.T

    return x_n, P_n, J


if __name__ == "__main__":
    import doctest

    doctest.testmod()
