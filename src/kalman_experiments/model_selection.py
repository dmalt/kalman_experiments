"""
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
from typing import Callable, NamedTuple, Sequence

import numpy as np
from scipy.optimize import nnls  # type: ignore
from tqdm import trange

from kalman_experiments.kalman.core import SimpleKF
from kalman_experiments.kalman.wrappers import PerturbedP1DMatsudaKF
from kalman_experiments.models import MatsudaParams
from kalman_experiments.numpy_types import Cov, Mat, Vec, Vec1D


class KFParams(NamedTuple):
    A: float
    f: float
    q_s: float
    r_s: float
    x_0: Vec
    P_0: Cov


def fit_kf_parameters(
    meas: Vec | Vec1D, KF: PerturbedP1DMatsudaKF, n_iter: int = 800, tol: float = 1e-3
) -> PerturbedP1DMatsudaKF:

    AMP_EPS = 1e-4
    sr = KF.mp.sr
    prev_freq = KF.mp.freq
    model_error = np.inf
    for _ in (pb := trange(n_iter, desc="Fitting KF parameters")):
        amp, freq, q_s, r_s, x_0, P_0 = em_step(meas, KF.KF, pb)
        amp = min(amp, 1 - AMP_EPS)
        freq *= sr / (2 * np.pi)

        mp = MatsudaParams(amp, freq, sr)
        KF = PerturbedP1DMatsudaKF(mp, q_s, KF.psi, r_s, KF.lambda_)
        KF.KF.x = x_0
        KF.KF.P = P_0
        model_error = abs(freq - prev_freq)
        prev_freq = freq
        if model_error < tol:
            break
    else:
        print(f"Did't converge after {n_iter} iterations; {model_error=}")
    return KF


def em_step(meas: Vec | Vec1D, KF: SimpleKF, pb) -> KFParams:
    n = len(meas)
    Phi, A, Q, R = KF.Phi, KF.H, KF.Q, KF.R
    assert n, "Measurements must be nonempty"

    y = normalize_measurement_dimensions(meas)
    x, P = apply_kf(KF, y)
    nll = compute_kf_nll(y, x, P, KF)
    x_n, P_n, J = apply_kalman_interval_smoother(KF, x, P)
    P_nt = estimate_adjacent_states_covariances(Phi, Q, A, R, P, J)

    S = compute_aux_em_matrices(x_n, P_n, P_nt)
    freq, Amp, q_s, r_s = params_update(S, Phi, n)
    pb.set_description(
        f"Fitting KF parameters: nll={nll:.2f},"
        f"f={freq*1000/2/np.pi:.2f}, A={Amp:.4f}, {q_s:.4f}, {r_s:.2f}"
    )
    x_0_new = x_n[0]
    P_0_new = P_n[0]

    return KFParams(Amp, freq, q_s, r_s, x_0_new, P_0_new)


def normalize_measurement_dimensions(meas: Vec1D) -> list[Vec]:
    # prepend nan for to simplify indexing; 0 index is for x and P prior to the measurements
    n = len(meas)
    y: list[Vec] = [np.array([[np.nan]])] * (n + 1)
    for t in range(1, n + 1):
        y[t] = meas[t - 1, np.newaxis, np.newaxis]
    return y


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


def estimate_adjacent_states_covariances(
    Phi: Mat, Q: Cov, A: Mat, R: Cov, P: list[Cov], J: list[Mat]
) -> list[Mat]:
    # estimate P^n_{t-1,t-2}
    n = len(P) - 1
    P_ = Phi @ P[n - 1] @ Phi.T + Q  # P^{n-1}_n
    K = np.linalg.solve(A @ P_ @ A.T + R, A @ P[n]).T  # K_n, eq. (6.23) in [1]
    P_nt: list[Cov] = [None] * (n + 1)  # type: ignore  # P^n_{t-1, t-2}
    P_nt[n - 1] = (np.eye(K.shape[0]) - K @ A) @ Phi @ P[n - 1]  # P^n_{n, n-1}, eq.(6.55) in [1]

    for t in range(n, 1, -1):
        P_nt[t - 2] = (
            P[t - 1] @ J[t - 2].T + J[t - 1] @ (P_nt[t - 1] - Phi @ P[t - 1]) @ J[t - 2].T
        )
    return P_nt


def compute_aux_em_matrices(x_n: list[Vec], P_n: list[Cov], P_nt: list[Mat]) -> dict[str, Mat]:
    n = len(x_n) - 1
    S = {
        "11": np.zeros_like(P_n[0], dtype=np.longdouble),
        "10": np.zeros_like(P_nt[0], dtype=np.longdouble),
        "00": np.zeros_like(P_n[0], dtype=np.longdouble),
    }
    for t in range(1, n + 1):
        S["11"] += x_n[t] @ x_n[t].T + P_n[t]
        S["10"] += x_n[t] @ x_n[t - 1].T + P_nt[t - 1]
        S["00"] += x_n[t - 1] @ x_n[t - 1].T + P_n[t - 1]
    return S


def params_update(S: dict[str, Mat], Phi: Mat, n: int) -> tuple[float, float, float, float]:
    A = S["00"][0, 0] + S["00"][1, 1]
    B = S["10"][0, 0] + S["10"][1, 1]
    C = S["10"][1, 0] - S["10"][0, 1]
    D = S["11"][0, 0] + S["11"][1, 1]
    f = max(C / B, 0)
    Amp = np.sqrt(B**2 + C**2) / A
    q_s = np.sqrt(max(0.5 * (D - Amp**2 * A) / n, 1e-6))
    r_s = np.sqrt(
        (S["11"][2, 2] - 2 * S["10"][2, :] @ Phi.T[:, 2] + (Phi[2, :] @ S["00"] @ Phi.T[:, 2])) / n
    )
    return float(f), float(Amp), float(q_s), float(r_s)


def compute_kf_nll(y: list[Vec], x: list[Vec], P: list[Cov], KF: SimpleKF) -> float:
    n = len(y) - 1
    negloglikelihood = 0
    for t in range(1, n + 1):
        x_, P_ = KF.predict(x[t], P[t])
        eps = y[t] - KF.H @ x_
        Sigma = KF.H @ P_ @ KF.H.T + KF.R
        tmp = np.linalg.solve(Sigma, eps)  # Sigma inversion
        negloglikelihood += 0.5 * (np.log(np.linalg.det(Sigma)) + eps.T @ tmp)
    return float(negloglikelihood)


def nll_opt_wrapper(x, meas, sr, psi, lambda_):
    A = x[0]
    f = x[1]
    q_s = x[2]
    r_s = x[3]
    y = normalize_measurement_dimensions(meas)
    mp = MatsudaParams(A, f, sr)
    # KF = PerturbedP1DMatsudaKF(KF.mp, q_s, KF.psi, r_s, KF.lambda_)
    KF = PerturbedP1DMatsudaKF(mp, q_s, psi, r_s, lambda_)
    x, P = apply_kf(KF.KF, y)
    return compute_kf_nll(y, x, P, KF.KF)


PsdFunc = Callable[[float], float]


def get_psd_val_from_est(f, freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Utility function to get estimated psd value by frequency

    If using welch for psd estimation, make sure it was called with
    `return_onesided=True` (default)
    """
    ind = np.argmin((freqs - f) ** 2)
    return psd[ind]


def estimate_sigmas(
    basis_psd_funcs: list[PsdFunc], data_psd_func: PsdFunc, freqs: Sequence[float]
) -> np.ndarray:
    A: list[list[float]] = []
    b = [1] * len(freqs)
    for row, f in enumerate(freqs):
        b_ = data_psd_func(f)
        A.append([])
        for func in basis_psd_funcs:
            A[row].append(func(f) / b_)
    return nnls(np.array(A), np.array(b))[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
