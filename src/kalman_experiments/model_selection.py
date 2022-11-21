"""
Examples
--------
Get smoothed data
>>> from kalman_experiments.kalman import PerturbedP1DMatsudaKF
>>> from kalman_experiments.models import MatsudaParams, SingleRhythmModel, collect
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> # Setup oscillatioins model and generate oscillatory signal
>>> mp = MatsudaParams(A=0.99, freq=10, sr=1000)
>>> gt_states = collect(SingleRhythmModel(mp, sigma=1), n_samp=1000)
>>> meas = np.real(gt_states) + 10*np.random.randn(len(gt_states))
>>> kf = PerturbedP1DMatsudaKF(mp, q_s=1, psi=np.zeros(0), r_s=10, lambda_=0).KF
>>> y = normalize_measurement_dimensions(meas)
>>> x, P = apply_kf(kf, y)
>>> x_n, P_n, J = apply_kalman_interval_smoother(kf, x, P)
>>> res = plt.plot([xx[0] for xx in x], label="fp", linewidth=4)
>>> res = plt.plot([xxn[0] for xxn in x_n], label="smooth", linewidth=4)
>>> res = plt.plot(np.real(gt_states), label="gt", linewidth=4)
>>> l = plt.legend()
>>> plt.show()

Fit params white noise
>>> from kalman_experiments.model_selection import fit_kf_parameters
>>> from kalman_experiments.kalman import PerturbedP1DMatsudaKF
>>> from kalman_experiments.models import MatsudaParams, SingleRhythmModel, collect
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> # Setup oscillatioins model and generate oscillatory signal
>>> mp = MatsudaParams(A=0.99, freq=10, sr=1000)
>>> gt_states = collect(SingleRhythmModel(mp, sigma=1), n_samp=1000)
>>> meas = np.real(gt_states) + 10*np.random.randn(len(gt_states))
>>> kf = PerturbedP1DMatsudaKF(MatsudaParams(A=0.999, freq=1, sr=1000), q_s=2, psi=np.zeros(0), r_s=5, lambda_=0)
>>> kf = fit_kf_parameters(meas, kf)
>>> print(kf.M)

Fit params pink noise
>>> from kalman_experiments.model_selection import fit_kf_parameters
>>> from kalman_experiments import SSPE
>>> from kalman_experiments.kalman import PerturbedP1DMatsudaKF
>>> from kalman_experiments.models import MatsudaParams, SingleRhythmModel, collect, gen_ar_noise_coefficients
>>> import numpy as np
>>> # Setup oscillatioins model and generate oscillatory signal
>>> sim = SSPE.gen_sine_w_pink(1, 1000)
>>> a = gen_ar_noise_coefficients(alpha=1, order=20)
>>> kf = PerturbedP1DMatsudaKF(MatsudaParams(A=0.8, freq=1, sr=1000), q_s=2, psi=a, r_s=1, lambda_=1e-3)
>>> kf = fit_kf_parameters(sim.data, kf)

"""
from typing import Callable, Collection, NamedTuple, Sequence

import numpy as np
from tqdm import trange

from kalman_experiments.kalman import PerturbedP1DMatsudaKF, SimpleKF
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
    sr = KF.M.sr
    prev_freq = KF.M.freq
    model_error = np.inf
    for _ in trange(n_iter, desc="Fitting KF parameters"):
        # Phi, Q, R, x_0, P_0 = em_step(meas, KF.KF, phi_full_upd, q_full_upd, r_full_upd)
        amp, freq, q_s, r_s, x_0, P_0 = em_step(meas, KF.KF, phi_osc_only_upd)
        # freq = np.arctan((Phi[1, 0] - Phi[0, 1]) / (Phi[0, 0] + Phi[1, 1])) / 2 / np.pi * sr
        # amp = min(
        #     np.sqrt(((Phi[1, 0] - Phi[0, 1]) ** 2 + (Phi[0, 0] + Phi[1, 1]) ** 2) / 4), 1 - AMP_EPS
        # )
        amp = min(amp, 1 - AMP_EPS)
        freq *= sr / (2 * np.pi)

        # amp = np.sqrt(((Phi[1, 0] - Phi[0, 1]) ** 2 + (Phi[0, 0] + Phi[1, 1]) ** 2) / 4)
        # q_s = np.sqrt((Q[0, 0] + Q[1, 1]) / 2)
        # r_s = np.sqrt(Q[2, 2])
        # psi = Phi[2, -len(KF.psi) : len(Phi)]
        KF = PerturbedP1DMatsudaKF(MatsudaParams(amp, freq, sr), q_s, KF.psi, r_s, KF.lambda_)
        # print(KF.M, KF.q_s, KF.r_s)
        KF.KF.x = x_0
        KF.KF.P = P_0
        model_error = abs(freq - prev_freq)
        prev_freq = freq
        if model_error < tol:
            break
    else:
        print(f"Did't converge after {n_iter} iterations; {model_error=}")
    return KF


PhiUpdateStrategy = Callable[[Mat, dict[str, Mat]], Mat]
QUpdateStrategy = Callable[[Cov, dict[str, Mat], Mat, int], Cov]
RUpdateStrategy = Callable[[Cov, Mat, list[Vec], list[Cov], list[Vec]], Cov]


def em_step(
    meas: Vec | Vec1D,
    KF: SimpleKF,
    phi_upd: PhiUpdateStrategy,
) -> KFParams:
    n = len(meas)
    Phi, A, Q, R = KF.Phi, KF.H, KF.Q, KF.R
    assert n, "Measurements must be nonempty"
    # assert meas.ndim == 2, "Measurements must be a column vector of shape (n_meas, 1)"

    y = normalize_measurement_dimensions(meas)
    x, P = apply_kf(KF, y)
    # print("nll, r2 =", compute_kf_negloglikelihood(y, x, P, KF))
    x_n, P_n, J = apply_kalman_interval_smoother(KF, x, P)
    P_nt = estimate_adjacent_states_covariances(Phi, Q, A, R, P, J)

    S = compute_aux_em_matrices(x_n, P_n, P_nt)
    freq, Amp, q_s, r_s = phi_upd(S, Phi, n)
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
    x: list[Vec] = [None] * (n + 1)  # pyright: ignore  # x^t_t
    P: list[Cov] = [None] * (n + 1)  # pyright: ignore  # P^t_t
    x[0], P[0] = KF.x, KF.P
    for t in range(1, n + 1):
        x[t], P[t] = KF.step(y[t])
    return x, P


def apply_kalman_interval_smoother(
    KF: SimpleKF, x: list[Vec], P: list[Cov]
) -> tuple[list[Vec], list[Cov], list[Mat]]:
    n = len(x) - 1
    x_n: list[Vec] = [None] * (n + 1)  # pyright: ignore  # x^n_t
    P_n: list[Cov] = [None] * (n + 1)  # pyright: ignore  # P^n_t
    x_n[n], P_n[n] = x[n], P[n]
    J: list[Mat] = [None] * (n + 1)  # pyright: ignore
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
    P_nt: list[Cov] = [None] * (n + 1)  # pyright: ignore  # P^n_{t-1, t-2}
    P_nt[n - 1] = (np.eye(K.shape[0]) - K @ A) @ Phi @ P[n - 1]  # P^n_{n, n-1}, eq.(6.55) in [1]

    for t in range(n, 1, -1):
        P_nt[t - 2] = (
            P[t - 1] @ J[t - 2].T + J[t - 1] @ (P_nt[t - 1] - Phi @ P[t - 1]) @ J[t - 2].T
        )
    return P_nt


def compute_aux_em_matrices(x_n: list[Vec], P_n: list[Cov], P_nt: list[Mat]) -> dict[str, Mat]:
    n = len(x_n) - 1
    S = {"11": np.zeros_like(P_n[0]), "10": np.zeros_like(P_nt[0]), "00": np.zeros_like(P_n[0])}
    for t in range(1, n + 1):
        S["11"] += x_n[t] @ x_n[t].T + P_n[t]
        S["10"] += x_n[t] @ x_n[t - 1].T + P_nt[t - 1]
        S["00"] += x_n[t - 1] @ x_n[t - 1].T + P_n[t - 1]
    return S


def phi_full_upd(Phi: Mat, S: dict[str, Mat]) -> Mat:
    return np.linalg.solve(S["00"], S["10"].T).T  # S_10 * S_["00"]^{-1}


def phi_osc_only_upd(S: dict[str, Mat], Phi: Mat, n: int) -> tuple[float, float, float, float]:
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
    return f, Amp, q_s, r_s


def q_full_upd(Q: Cov, S: dict[str, Mat], Phi_: Mat, n: int) -> Cov:
    return (S["11"] - S["10"] @ Phi_.T - Phi_ @ S["10"].T + Phi_ @ S["00"] @ Phi_.T) / n


def r_full_upd(R: Cov, A: Mat, x_n: list[Vec], P_n: list[Cov], y: list[Vec]) -> Cov:
    n, sensors_cnt = len(x_n) - 1, A.shape[0]
    res = np.zeros((sensors_cnt, sensors_cnt))
    for t in range(1, n + 1):
        tmp = y[t] - A @ x_n[t]
        res += tmp @ tmp.T + A @ P_n[t] @ A.T
    return res / n


def r_null_upd(R: Cov, A: Mat, x_n: list[Vec], P_n: list[Cov], y: list[Vec]) -> Cov:
    return np.zeros_like(R)


def compute_kf_negloglikelihood(
    y: list[Vec], x: list[Vec], P: list[Cov], KF: SimpleKF
) -> tuple[float, float]:
    n = len(y) - 1
    negloglikelihood = 0
    r_2: float = 0
    for t in range(1, n + 1):
        x_, P_ = KF.predict(x[t], P[t])
        eps = y[t] - KF.H @ x_
        r_2 += float(eps @ eps.T)
        Sigma = KF.H @ P_ @ KF.H.T + KF.R
        tmp = np.linalg.solve(Sigma, eps)  # Sigma inversion
        negloglikelihood += 0.5 * (np.log(np.linalg.det(Sigma)) + eps.T @ tmp)
    return negloglikelihood, r_2


def theor_psd_ar(f: float, s: float, ar_coef: Collection[float], fs: float) -> float:
    denom = 1 - sum(a * np.exp(-2j * np.pi * f / fs * m) for m, a in enumerate(ar_coef, 1))
    return s**2 / np.abs(denom) ** 2


def theor_psd_mk_mar(f: float, s: float, mp: MatsudaParams) -> float:
    """Theoretical PSD for Matsuda-Komaki multivariate AR process"""
    phi = 2 * np.pi * mp.freq / mp.sr
    psi = 2 * np.pi * f / mp.sr
    A = mp.A

    denom = np.abs(1 - 2 * A * np.cos(phi) * np.exp(-1j * psi) + A**2 * np.exp(-2j * psi)) ** 2
    num = 1 + A**2 - 2 * A * np.cos(phi) * np.cos(psi)
    return s**2 * num / denom


PsdFunc = Callable[[float], float]


def get_psd_val_from_est(f, freqs: np.ndarray, psd: np.ndarray) -> float:
    ind = np.argmin((freqs - f) ** 2)
    return psd[ind]


def estimate_sigmas(
    basis_psd_funcs: list[PsdFunc], data_psd_func: PsdFunc, freqs: Sequence[float]
) -> np.ndarray:
    A = []
    b = [1] * len(freqs)
    for row, f in enumerate(freqs):
        b_ = data_psd_func(f)
        A.append([])
        for func in basis_psd_funcs:
            A[row].append(func(f) / b_)
    return np.linalg.lstsq(np.array(A), np.array(b))[0]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
