# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from functools import partial

import numpy as np
import pylab as plt
import scipy.signal as sg
from sklearn.metrics import roc_auc_score, roc_curve

from kalman_experiments.kalman.wrappers import (
    PerturbedP1DMatsudaKF,
    PerturbedP1DMatsudaSmoother,
    apply_kf,
)
from kalman_experiments.model_selection import (
    estimate_sigmas,
    fit_kf_parameters,
    get_psd_val_from_est,
    theor_psd_ar,
    theor_psd_mk_mar,
)
from kalman_experiments.models import MatsudaParams, gen_ar_noise_coefficients

# %%
data = np.load("/home/altukhov/Code/python/cfir-kalman/data/eegbci2_ICA_real_feet_fist.npz")
# np.random.seed(42)

delay = 0

# load data balanced
events = data["events"]
eeg = data["eeg"]
labels = data["labels"]
SRATE = data["fs"]
print(SRATE)
selected = []
for ind, (ev1, ev2) in enumerate(zip(events[:-1], events[1:])):
    l1, l2 = ev1[2], ev2[2]
    if l1 == 3:
        selected.append(ind)
        selected.append(ind + 1)

eeg_events = []
label_events = []
for k in range(len(events) - 1):
    if k in selected:
        eeg_events.append(eeg[events[k, 0] : events[k + 1, 0]])
        label_events.append(labels[events[k, 0] : events[k + 1, 0]])

eeg = np.concatenate(eeg_events)
labels = np.concatenate(label_events)
labels = labels == 1
print(SRATE)


def to_db(arr):
    return 10 * np.log10(arr)


def fit_parameters(
    train_data: np.ndarray,
    mp_init: MatsudaParams,
    noise_alpha: float,
    noise_order: int,
    nperseg: int,
) -> PerturbedP1DMatsudaKF:
    fit_freqs = [5, f0 * 2.1, 69]
    freqs, psd = sg.welch(train_data, fs=mp_init.sr, nperseg=nperseg)
    est_psd_func = partial(get_psd_val_from_est, freqs=freqs, psd=psd / 2)
    psi = gen_ar_noise_coefficients(noise_alpha, noise_order)
    ar_psd_func = partial(theor_psd_ar, ar_coef=psi, sr=SRATE, s=1)
    mar_psd_func = partial(theor_psd_mk_mar, s=1, mp=mp_init)

    q_s_2, r_s_2 = estimate_sigmas([mar_psd_func, ar_psd_func], est_psd_func, fit_freqs)
    q_s_est, r_s_est = np.sqrt(q_s_2 * SRATE), np.sqrt(r_s_2 * SRATE)
    r_s_est, q_s_est = max(r_s_est, 0.0001), max(q_s_est, 0.0001)

    r_s_est = np.sqrt(np.var(train_data) * (1 - 0.999**2) * 2)
    r_s_2 = r_s_est ** 2 / SRATE
    q_s_est = 0.1
    q_s_2 = q_s_est ** 2 / SRATE

    freqs_plt = freqs
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_plt, [to_db(est_psd_func(f)) for f in freqs_plt], label="estimated")
    plt.plot(freqs_plt, [to_db(mar_psd_func(f) * q_s_2) for f in freqs_plt], label="mar")

    plt.plot(
        freqs_plt,
        [to_db(mar_psd_func(f) * q_s_2 + ar_psd_func(f) * r_s_2) for f in freqs_plt],
        label="sum",
    )
    plt.legend()
    plt.show()
    print(f"{q_s_est=:.2e}, {r_s_est=:.2e}")

    psi = gen_ar_noise_coefficients(alpha=noise_alpha, order=noise_order)
    kf = PerturbedP1DMatsudaKF(mp_init, q_s=q_s_est, psi=psi, r_s=r_s_est)
    # kf = PerturbedP1DMatsudaKF(mp_init, q_s=q_s_est, psi=np.array([0.999]), r_s=r_s_est)
    # return fit_kf_parameters(train_data, kf, tol=1e-6)
    return kf


ff, psd = sg.welch(eeg, fs=SRATE, nperseg=1000)
plt.figure()
plt.plot(ff, to_db(psd))
alpha = 1
plt.plot(ff, [to_db(0.1 / f**alpha) for f in ff])
plt.show()


# process parameters
band = np.array([9, 13])
f0 = float(np.mean(band))

mp = MatsudaParams(A=0.99, freq=f0 * 2.1, sr=SRATE)

kf_aug = fit_parameters(eeg, mp, noise_alpha=alpha, noise_order=100, nperseg=2000)
# print(kf_aug)


Psi = 0.999
r_s_diff = np.sqrt(np.var(eeg) * (1 - Psi**2) * 2)
print(f"{r_s_diff=}")
kf_diff = PerturbedP1DMatsudaKF(mp, q_s=0.1, r_s=r_s_diff, psi=np.array([0]))
# kf_diff = PerturbedP1DMatsudaKF(mp, q_s=0.1, r_s=r_s_diff, psi=np.array([0]))
# kf_aug = PerturbedP1DMatsudaKF(mp, q_s=0.1, r_s=np.sqrt(np.var(eeg) * (1 - Psi**2) * 2), psi=a)


def smooth(x, m=SRATE):
    b = np.ones(m) / m
    return sg.lfilter(b, [1.0], x)


eeg_filt_diff = apply_kf(signal=eeg, kf=kf_diff, delay=0)
eeg_filt_aug = apply_kf(signal=eeg, kf=kf_aug, delay=0)

eeg_env_diff = smooth(np.abs(eeg_filt_diff))
eeg_env_aug = smooth(np.abs(eeg_filt_aug))


plt.plot(labels, label="label")
plt.plot(eeg_env_diff, label="diff kf")
plt.plot(eeg_env_aug, label="aug kf")
plt.legend()

plt.figure()
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

plt.plot(
    *roc_curve(labels, eeg_env_diff)[:2],
    label="diff kf auc = {:.2f}".format(roc_auc_score(labels, eeg_env_diff)),
)
plt.plot(
    *roc_curve(labels, eeg_env_aug)[:2],
    label="aug kf auc = {:.2f}".format(roc_auc_score(labels, eeg_env_aug)),
)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
# plt.plot(
#     *roc_curve(labels, skf_envelope)[:2],
#     label="KF auc = {:.2f}".format(roc_auc_score(labels, skf_envelope))
# )
# plt.plot(
#     *roc_curve(labels, ckf_envelope)[:2],
#     label="CKF auc = {:.2f}".format(roc_auc_score(labels, ckf_envelope))
# )
# plt.plot(
#     *roc_curve(labels, flkf_envelope)[:2],
#     label="FLKF auc = {:.2f}".format(roc_auc_score(labels, flkf_envelope))
# )
