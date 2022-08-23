import numpy as np

Arr = np.ndarray


class ColoredMeasurementNoiseKF:
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

    def __init__(self, n_x: int, n_z: int, Phi: Arr, Q: Arr, H: Arr, Psi: Arr, R: Arr):
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.Psi = Psi
        self.R = R

        self.x_posterior = np.zeros(n_x)[:, np.newaxis]
        self.P_posterior = np.zeros((n_x, n_x))

        self.ym1 = np.zeros(n_z)

    def predict(self):
        x_prior = self.Phi @ self.x_posterior  # eq. (26) from [1]
        P_prior = self.Phi @ self.P_posterior @ self.Phi.T + self.Q  # eq. (27) from [1]
        return x_prior, P_prior

    def update(self, y):
        x_prior, P_prior = self.predict()

        z = y - self.Psi * self.ym1  # eq. (35) from [1]
        n = z - self.H @ x_prior + self.Psi * self.H @ self.x_posterior  # eq. (37) from [1]
        # eq. (38) from [1]
        Sigma = (
            self.H @ P_prior @ self.H.T
            + self.Psi @ self.H @ self.P_posterior @ self.H.T @ self.Psi.T
            + self.R
            - self.H @ self.Phi @ self.P_posterior @ self.H.T @ self.Psi.T
            - self.Psi @ self.H @ self.P_posterior @ self.Phi.T @ self.H.T
        )
        # eq. (39) from [1]
        Pxn = P_prior @ self.H.T - self.Phi @ self.P_posterior @ self.H.T @ self.Psi.T

        K = Pxn / Sigma  # eq. (40) from [1]
        self.x_posterior = x_prior + K * n  # eq. (41) from [1]
        # eq. (42) from [1]
        self.P_posterior = P_prior - K * Sigma @ K.T
        self.ym1 = y
        return x_prior, self.x_posterior, n, K, Sigma, self.P_posterior


def plv(x1, x2, ma_len_samp):
    x1 /= np.abs(x1)
    x2 /= np.abs(x2)
    prod = np.conj(x1) * x2

    ma_kernel = np.ones(ma_len_samp) / ma_len_samp
    return np.convolve(prod, ma_kernel, mode="same"), prod[: len(prod) // 2].mean()


def env_cor(x1, x2, ma_len_samp):
    x1 = np.abs(x1)
    x1 -= x1.mean()
    x1 /= x1.std()
    x2 = np.abs(x2)
    x2 -= x2.mean()
    x2 /= x2.std()
    prod = x1 * x2

    ma_kernel = np.ones(ma_len_samp) / ma_len_samp
    return np.convolve(prod, ma_kernel, mode="same"), prod.mean()


if __name__ == "__main__":
    from gen_ar_noise import ArNoise
    from models import SingleRhythmModel, complex2mat, vec2complex
    from real_noise import prepare_real_noise

    order = 1
    alpha = 1
    sigma_noise = 4
    meas_noise = ArNoise(y0=np.random.rand(order), alpha=alpha, order=order, sigma=sigma_noise)
    # meas_noise = prepare_real_noise(raw_path="./sub-01_ses-session2_task-eyesopen_eeg.vhdr")
    sr = 125
    sigma_data = 1
    model = SingleRhythmModel(freq=10, A=0.99, sigma=sigma_data, sr=sr, meas_noise_ar=meas_noise)

    kf = ColoredMeasurementNoiseKF(
        n_x=2,
        n_z=1,
        Phi=complex2mat(model.Phi),
        Q=np.eye(N=2) * sigma_data**2,
        H=np.array([1, 0])[np.newaxis, :],
        # Psi=np.array([meas_noise.a[0]])[:, np.newaxis],
        Psi=np.array([-0.5])[:, np.newaxis],
        R=np.array([(sigma_noise) ** 2])[:, np.newaxis],
    )
    gt_states = []
    kf_states = []
    meas = []
    for i in range(1000):
        y = model.step()
        gt_states.append(model.x)
        meas.append(y)
        kf_states.append(vec2complex(kf.update(y)[1][:, 0]))
    for i in range(1000):
        y = model.step()
        gt_states.append(model.x)
        pred = kf.predict()
        kf.x_posterior = pred[0]
        kf.P_posterior = pred[1]
        kf_states.append(vec2complex(pred[0][:, 0]))
        meas.append(y)
    gt_states = np.array(gt_states)
    kf_states = np.array(kf_states)
    meas = np.array(meas)

    from cfir import CFIRBandDetector

    cfir = CFIRBandDetector([8, 12], sr, delay=0)
    cfir_states = cfir.apply(meas)

    import matplotlib.pyplot as plt

    lim = 2000
    t = np.arange(lim) / sr - lim / 2 / sr
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, np.real(cfir_states[:lim]))
    ax1.plot(t, np.real(kf_states[:lim]))
    ax1.plot(t, meas[:lim], alpha=0.3)
    ax1.plot(t, np.real(gt_states[:lim]), "--")
    ax1.legend(["cfir state (Re)", "kalman state (Re)", "measurements", "ground truth state (Re)"])
    ax1.grid()

    # ax1.grid()
    plt.xlabel("Time, sec")

    plv_win_kf, plv_tot_kf = plv(gt_states.copy(), kf_states.copy(), int(0.5 * sr))
    plv_win_cfir, plv_tot_cfir = plv(gt_states.copy(), cfir_states.copy(), int(0.5 * sr))
    envcor_win_kf, envcor_tot_kf = env_cor(gt_states.copy(), kf_states.copy(), int(0.5 * sr))
    envcor_win_cfir, envcor_tot_cfir = env_cor(gt_states.copy(), cfir_states.copy(), int(0.5 * sr))
    print(f"{np.abs(plv_tot_kf)=}, {np.abs(plv_tot_cfir)=}")
    ax2.plot(t, np.abs(plv_win_cfir), linewidth=2)
    ax2.plot(t, np.abs(plv_win_kf), linewidth=2)
    # ax2.plot(t, envcor_win_kf, linewidth=2)
    # ax2.plot(t, envcor_win_cfir, linewidth=2)
    ax2.legend(["plv(gt, cfir)", "plv(gt, kf)"])  # , "envcor(gt, kf)", "envcor(gt, cfir)"])
    plt.grid()
    plt.show()
