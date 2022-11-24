"""
Examples
--------
Get smoothed data
>>> from kalman_experiments.kalman import SossAugSmoother
>>> from kalman_experiments.models import MatsudaParams, SingleRhythmModel, collect
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> # Setup oscillatioins model and generate oscillatory signal
>>> mp = MatsudaParams(A=0.99, freq=10, sr=1000)
>>> gt_states = collect(SingleRhythmModel(mp, sigma=1), n_samp=1000)
>>> meas = np.real(gt_states) + 10*np.random.randn(len(gt_states))
>>> kf = SossAugSmoother(mp, q_s=1, psi=np.zeros(0), r_s=10, lambda_=0).KF
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
>>> from kalman_experiments.kalman import SossAugSmoother
>>> from kalman_experiments.models import MatsudaParams, SingleRhythmModel, collect
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> # Setup oscillatioins model and generate oscillatory signal
>>> mp = MatsudaParams(A=0.99, freq=10, sr=1000)
>>> gt_states = collect(SingleRhythmModel(mp, sigma=1), n_samp=1000)
>>> meas = np.real(gt_states) + 10*np.random.randn(len(gt_states))
>>> mp = MatsudaParams(A=0.999, freq=1, sr=1000)
>>> kf = SossAugSmoother(mp, q_s=2, psi=np.zeros(0), r_s=5, lambda_=0)
>>> kf = fit_kf_parameters(meas, kf)
>>> print(kf.M)

Fit params pink noise
>>> from kalman_experiments.model_selection import fit_kf_parameters
>>> from kalman_experiments import SSPE
>>> from kalman_experiments.kalman import SossAugSmoother
>>> from kalman_experiments.models import (
>>>     MatsudaParams, SingleRhythmModel, collect, gen_ar_noise_coefficients
>>> )
>>> import numpy as np
>>> # Setup oscillatioins model and generate oscillatory signal
>>> sim = SSPE.gen_sine_w_pink(1, 1000)
>>> a = gen_ar_noise_coefficients(alpha=1, order=20)
>>> mp = MatsudaParams(A=0.8, freq=1, sr=1000)
>>> kf = SossAugSmoother(mp, q_s=2, psi=a, r_s=1, lambda_=1e-3)
>>> kf = fit_kf_parameters(sim.data, kf)

"""
from dataclasses import dataclass
from typing import Callable, Collection, Sequence

import numpy as np
from tqdm import trange

from kalman_experiments.complex import complex2mat
from kalman_experiments.kalman.core import Gaussian, SimpleKF
from kalman_experiments.kalman.onedim import SossAugSmoother
from kalman_experiments.models import MatsudaParams
from kalman_experiments.numpy_types import (
    Cov,
    Mat,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Vec,
    Vec1D,
    check_positive_float,
    check_positive_int,
)


@dataclass
class KFParams:
    A: NonNegativeFloat
    freq_rad: NonNegativeFloat
    q_s: NonNegativeFloat
    r_s: NonNegativeFloat


def fit_kf_parameters(
    meas: Vec | Vec1D,
    KF: SossAugSmoother,
    n_iter: PositiveInt = check_positive_int(800),
    tol: PositiveFloat = check_positive_float(1e-3),
) -> SossAugSmoother:

    AMP_EPS = 1e-4
    sr = KF.mp.sr
    prev_freq_hz = KF.mp.freq_hz
    model_error = np.inf
    for i in trange(n_iter, desc="Fitting KF parameters"):
        S, prior_state = em_step_general(meas, KF.KF)
        kfp = params_update(S, KF, check_positive_int(len(meas)))

        amp = check_positive_float(min(kfp.A, 1 - AMP_EPS))
        # amp = kfp.A
        freq_hz = check_positive_float(kfp.freq_rad * sr / (2 * np.pi))
        mp = MatsudaParams(amp, freq_hz, sr)

        # kfp.r_s = 1
        KF = SossAugSmoother(mp, kfp.q_s, KF.psi, kfp.r_s, lambda_=KF.lambda_)
        KF.KF.state = prior_state
        model_error = abs(freq_hz - prev_freq_hz)
        prev_freq_hz = freq_hz
        if model_error < tol:
            break
    else:
        print(f"Did't converge after {n_iter} iterations; {model_error=}")
    return KF


KFParamsUpdStrategy = Callable[[dict[str, Mat], Mat, PositiveInt], KFParams]


def em_step_general(meas: Vec1D, KF: SimpleKF) -> tuple[dict[str, Mat], Gaussian]:
    Phi, H, Q, R = KF.Phi, KF.H, KF.Q, KF.R

    y = normalize_measurement_dimensions(meas)
    # x, P = apply_kf(KF, y)
    states_fp = KF.apply(y)
    assert not any(np.any(np.isnan(s.x)) or np.any(np.isnan(s.P)) for s in states_fp)
    assert not any(np.linalg.norm(s.P) > 1e10 for s in states_fp)
    # assert False
    print("nll = ", compute_kf_nll(y, states_fp, KF))
    states_n, J = apply_kalman_interval_smoother(KF, states_fp)
    P_nt = estimate_adj_state_cross_covariances(Phi, Q, H, R, [s.P for s in states_fp], J)
    assert not any(np.linalg.norm(P) > 1e10 for P in P_nt)

    S = compute_aux_em_matrices(states_n, P_nt)
    # return kf_upd_func(S, Phi, n), Gaussian(x_n[0], P_n[0])
    assert not np.any(np.isnan(S["00"]))
    assert not np.any(np.isnan(S["10"]))
    assert not np.any(np.isnan(S["11"]))
    return S, states_n[0]


def normalize_measurement_dimensions(meas: Vec1D) -> list[Vec]:
    return [np.array([[m]]) for m in meas]


def apply_kalman_interval_smoother(
    KF: SimpleKF, states_fp: list[Gaussian]
) -> tuple[list[Gaussian], list[Mat]]:
    n = len(states_fp) - 1
    # x_n: list[Vec] = [np.zeros_like(states_fp[0].x)] * (n + 1)  # pyright: ignore  # x^n_t
    # P_n: list[Cov] = [np.zeros_like(states_fp[0].P)] * (n + 1)  # pyright: ignore  # P^n_t
    states_n = [Gaussian(np.zeros_like(states_fp[0].x), np.zeros_like(states_fp[0].P))] * (n + 1)
    states_n[n] = states_fp[n]
    J: list[Mat] = [np.empty_like(states_fp[0].P)] * (n + 1)  # pyright: ignore
    for t in range(n, 0, -1):
        states_n[t - 1], J[t - 1] = smoother_step(KF, states_fp[t - 1], states_n[t])

    return states_n, J


def smoother_step(KF: SimpleKF, state: Gaussian, state_n: Gaussian) -> tuple[Gaussian, Mat]:
    """
    Make one Kalman Smoother step


    TODO: Update docstring

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
        Smoothed state covariance for one timestep back, i.e. P^{n}_{t-1} in
        eq. (6.48) in [1]
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
    state_ = KF.predict(state)

    # P * Phi^T * P_^{-1}; solve is better than inv
    J = np.linalg.solve(state_.P, KF.Phi @ state.P).T

    x = state.x + J @ (state_n.x - state_.x)
    P = state.P + J @ (state_n.P - state_.P) @ J.T
    state_n = Gaussian(x, P)

    return state_n, J


def estimate_adj_state_cross_covariances(
    Phi: Mat, Q: Cov, H: Mat, R: Cov, all_P: list[Cov], J: list[Mat]
) -> list[Mat]:
    # estimate P^n_{t-1,t-2}
    n = len(all_P) - 1
    m_sen = H.shape[0]
    P_ = Phi @ all_P[n - 1] @ Phi.T + Q  # P^{n-1}_n
    K = np.linalg.solve(H @ P_ @ H.T + R, H @ all_P[n]).T  # K_n, eq. (6.23) in [1]
    P_nt: list[Cov] = [np.zeros_like(P_)] * (n + 1)  # pyright: ignore  # P^n_{t-1, t-2}
    P_nt[n - 1] = (np.eye(m_sen) - K @ H) @ Phi @ all_P[n - 1]  # P^n_{n, n-1}, eq.(6.55) in [1]

    for t in range(n, 1, -1):
        P_nt[t - 2] = (
            all_P[t - 1] @ J[t - 2].T + J[t - 1] @ (P_nt[t - 1] - Phi @ all_P[t - 1]) @ J[t - 2].T
        )
        assert np.linalg.norm(P_nt[t - 2]) < 1e10
    return P_nt


def compute_aux_em_matrices(states_n: list[Gaussian], P_nt: list[Mat]) -> dict[str, Mat]:
    n = len(states_n) - 1
    S = {
        "11": np.zeros_like(states_n[0].P),
        "10": np.zeros_like(P_nt[0]),
        "00": np.zeros_like(states_n[0].P),
    }
    for t in range(1, n + 1):
        S["11"] += states_n[t].x @ states_n[t].x.T + states_n[t].P
        S["10"] += states_n[t].x @ states_n[t - 1].x.T + P_nt[t - 1]
        S["00"] += states_n[t - 1].x @ states_n[t - 1].x.T + states_n[t - 1].P
    return S


def params_update(S: dict[str, Mat], KF: SossAugSmoother, n: PositiveInt) -> KFParams:
    A, B = S["00"][0, 0] + S["00"][1, 1], S["10"][0, 0] + S["10"][1, 1]
    C, D = S["10"][1, 0] - S["10"][0, 1], S["11"][0, 0] + S["11"][1, 1]
    # Phi = KF.KF.Phi.copy()
    freq_rad = max(C / B, 0)
    Amp = np.sqrt(B**2 + C**2) / A
    # mp = MatsudaParams(A=Amp, freq_hz=freq_rad / 2 / np.pi * KF.mp.sr, sr=KF.mp.sr)
    # Phi_small = complex2mat(mp.Phi)
    # Phi[:2, :2] = Phi_small
    q_s = np.sqrt(max(0.5 * (D - Amp**2 * A) / n, 1e-6))
    # r_s_2 = S["11"][2, 2] - 2 * S["10"][2, :] @ Phi.T[:, 2] + Phi[2, :] @ S["00"] @ Phi.T[:, 2]
    r_s_2 = S["11"][2, 2] - (S["10"] @ np.linalg.solve(S["00"], S["10"].T))[2, 2]
    # assert r_s_2 >= 0
    r_s = np.sqrt(max(r_s_2 / n, 1e-6))
    print(f"{Amp=}, f={freq_rad/ 2/ np.pi * 500}, {q_s=}, {r_s=}")
    return KFParams(Amp, freq_rad, q_s, r_s)


def wrapper(x, sr, meas, psi, A, f):
    # A = x[0]
    # f = x[1]
    q_s = x[0]
    r_s = x[1]
    mp = MatsudaParams(A, f, sr)
    y = normalize_measurement_dimensions(meas)

    # kfp.r_s = 1
    KF = SossAugSmoother(mp, q_s, psi, r_s, lambda_=0)
    states_fp = KF.KF.apply(y)
    return compute_kf_nll(y, states_fp, KF.KF)


def compute_kf_nll(y: list[Vec], states: list[Gaussian], KF: SimpleKF) -> float:
    n = len(y) - 1
    negloglikelihood = 0
    r_2: float = 0
    for t in range(1, n + 1):
        state_ = KF.predict(states[t])
        eps = y[t] - KF.H @ state_.x
        r_2 += float(eps @ eps.T)
        Sigma = KF.H @ state_.P @ KF.H.T + KF.R
        tmp = np.linalg.solve(Sigma, eps)  # Sigma inversion
        negloglikelihood += 0.5 * (np.log(np.linalg.det(Sigma)) + eps.T @ tmp)
    return float(negloglikelihood)


def theor_psd_ar(
    f: PositiveFloat, s: NonNegativeFloat, ar_coef: Collection[float], sr: PositiveFloat
) -> PositiveFloat:
    denom = 1 - sum(a * np.exp(-2j * np.pi * f / sr * m) for m, a in enumerate(ar_coef, 1))
    return check_positive_float(s**2 / np.abs(denom) ** 2)


def theor_psd_mk_mar(f: PositiveFloat, s: NonNegativeFloat, mp: MatsudaParams) -> PositiveFloat:
    """Theoretical PSD for Matsuda-Komaki multivariate AR process"""
    phi = 2 * np.pi * mp.freq_hz / mp.sr
    psi = 2 * np.pi * f / mp.sr
    A = mp.A

    denom = np.abs(1 - 2 * A * np.cos(phi) * np.exp(-1j * psi) + A**2 * np.exp(-2j * psi)) ** 2
    num = 1 + A**2 - 2 * A * np.cos(phi) * np.cos(psi)
    return check_positive_float(s**2 * num / denom)


PsdFunc = Callable[[PositiveFloat], PositiveFloat]


def get_psd_val_from_est(f: PositiveFloat, freqs: np.ndarray, psd: np.ndarray) -> PositiveFloat:
    ind = np.argmin((freqs - f) ** 2)
    return check_positive_float(psd[ind])


def estimate_sigmas_squared(
    basis_psd_funcs: Sequence[PsdFunc], data_psd_func: PsdFunc, freqs: Sequence[PositiveFloat]
) -> list[PositiveFloat]:
    A: list[list[float]] = []
    b = [1.0] * len(freqs)
    for row, f in enumerate(freqs):
        b_ = data_psd_func(f)
        A.append([])
        for func in basis_psd_funcs:
            A[row].append(func(f) / b_)
    return [
        check_positive_float(s) for s in np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    ]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
