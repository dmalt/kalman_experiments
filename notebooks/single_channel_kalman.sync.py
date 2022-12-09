# In[1]:
import matplotlib.pyplot as plt
import numpy as np

from kalman_experiments.cfir import CFIRParams, apply_cfir
from kalman_experiments.kalman.wrappers import PerturbedP1DMatsudaSmoother, apply_kf
from kalman_experiments.metrics import crosscorr, env_cor, plv
from kalman_experiments.models import (
    ArNoiseModel,
    MatsudaParams,
    SingleRhythmModel,
    collect,
    prepare_real_noise,
)
from kalman_experiments.plots import (
    plot_crosscorrelations,
    plot_generated_signal,
    plot_kalman_vs_cfir,
)

# # Simulated AR noise
# Parameter A in the code below corresponds to A in Matsuda's model:
# $$x_{k+1} = A e^{\frac{2 \pi i f}{sr}} x_k + \nu, \nu \sim N(0, \sigma ^ 2)$$

# In[2]:


SRATE = 500
N_SAMP = 10_000


# In[3]:


# Setup oscillatioins model and generate oscillatory signal
FREQ_GT = 10
A_GT = 0.99  # as in x_next = A*exp(2*pi*OSCILLATION_FREQ / sr)
SIGNAL_SIGMA_GT = 1  # std of the model-driving white noise in the Matsuda model

mp = MatsudaParams(A_GT, FREQ_GT, SRATE)
oscillation_model = SingleRhythmModel(mp, cont_sigma=SIGNAL_SIGMA_GT)
gt_states = collect(oscillation_model, N_SAMP)


# In[4]:


# Setup simulated noise and measurements
NOISE_AR_ORDER = 30
ALPHA = 1
SIM_NOISE_SIGMA_GT = 1  # std of white noise driving the ar model for the colored noise

noise_model = ArNoiseModel(
    x0=np.random.rand(NOISE_AR_ORDER), alpha=ALPHA, order=NOISE_AR_ORDER, s=SIM_NOISE_SIGMA_GT
)
noise_sim = collect(noise_model, N_SAMP)
meas = np.real(gt_states)  # + noise_sim


# In[5]:


# Plot generated signal
legend = [
    "Generated signal",
    f"$1/f^{ {ALPHA} }$",
    f"AR({NOISE_AR_ORDER})" f" for $1/f^{ {ALPHA} }$ noise",
]
plot_generated_signal(noise_sim, meas, sr=SRATE, alpha=ALPHA, legend=legend, tmin=0, tmax=2)
plt.show()


# In[6]:


# Setup filters

A_KF = A_GT
FREQ_KF = FREQ_GT
SIGNAL_SIGMA_KF = SIGNAL_SIGMA_GT
# PSI = 0
PSI = 0.5
# PSI = -0.5
NOISE_SIGMA_KF = SIM_NOISE_SIGMA_GT
DELAY = -10
DELAY_CFIR = DELAY

# kf = Difference1DMatsudaKF(A=A_KF, f=FREQ_KF, sr=SRATE, q_s=SIGNAL_SIGMA_KF, psi=PSI, r_s=NOISE_SIGMA_KF)
# kf = PerturbedP1DMatsudaKF(
#     A=A_KF, f=FREQ_KF, sr=SRATE, q_s=SIGNAL_SIGMA_KF, psi=noise_model.a, r_s=NOISE_SIGMA_KF, lambda_=0
# )
mp = MatsudaParams(A_KF, FREQ_KF, SRATE)
kf = PerturbedP1DMatsudaSmoother(
    mp, q_s=SIGNAL_SIGMA_KF, psi=noise_model.a, r_s=NOISE_SIGMA_KF, lag=5, lambda_=0
)
cfir = CFIRParams((8, 12), SRATE)


# In[7]:


# Filter measurements with simulated noise

cfir_states = apply_cfir(cfir, meas, delay=DELAY_CFIR)
kf_states = apply_kf(kf, meas, delay=DELAY)


# In[8]:


# Plot results for simulated noise

kf_delayed = np.roll(kf_states, shift=-DELAY)
cfir_delayed = np.roll(cfir_states, shift=-DELAY)

plv_win_kf, plv_tot_kf = plv(gt_states, kf_delayed.copy(), int(0.5 * SRATE))
plv_win_cfir, plv_tot_cfir = plv(gt_states, cfir_delayed.copy(), int(0.5 * SRATE))
envcor_kf = env_cor(gt_states.copy(), kf_delayed.copy())
envcor_cfir = env_cor(gt_states.copy(), cfir_delayed.copy())
print(
    "KF total PLV = ",
    round(np.abs(plv_tot_kf), 2),
    "CFIR total PLV = ",
    round(np.abs(plv_tot_cfir), 2),
    end=" ",
)
print("KF envcor = ", round(envcor_kf, 2), "CFIR envcor = ", round(envcor_cfir, 2))

plot_kalman_vs_cfir(
    meas, gt_states, kf_states, cfir_states, plv_win_kf, plv_win_cfir, 1000, SRATE, DELAY
)
plt.show()

# Plot envelope cross-correlations and delays

t, corrs_cfir = crosscorr(np.abs(gt_states), np.abs(cfir_states), SRATE, 150)
t, corrs_kf = crosscorr(np.abs(gt_states), np.abs(kf_states), SRATE, 150)
t_ms = t * 1000
res = plot_crosscorrelations(t_ms, corrs_cfir, corrs_kf)
plt.show()


# # Real noise

# In[9]:


# Setup real noise and generate measurements
REAL_NOISE_SIGMA_GT = SIM_NOISE_SIGMA_GT + 3

raw_path = "./data/ds004148/sub-01/ses-session2/eeg/sub-01_ses-session2_task-eyesopen_eeg.vhdr"
real_noise_model, srate = prepare_real_noise(raw_path=raw_path, s=REAL_NOISE_SIGMA_GT)

noise_real = collect(real_noise_model, N_SAMP)
meas = np.real(gt_states) + noise_real

legend = ["Generated signal", f"1/f", "Real noise"]
plot_generated_signal(noise_real, meas, sr=int(srate), alpha=ALPHA, tmin=0, tmax=2, legend=legend)
plt.show()


# In[10]:


DELAY = -20
cfir_states = apply_cfir(cfir, meas, delay=DELAY)
kf_states = apply_kf(kf, meas, delay=DELAY)


# In[11]:


kf_delayed = np.roll(kf_states, shift=-DELAY)
cfir_delayed = np.roll(cfir_states, shift=-DELAY)

plv_win_kf, plv_tot_kf = plv(gt_states.copy(), kf_delayed.copy(), int(0.5 * SRATE))
plv_win_cfir, plv_tot_cfir = plv(gt_states.copy(), cfir_delayed.copy(), int(0.5 * SRATE))

plot_kalman_vs_cfir(
    meas, gt_states, kf_states, cfir_states, plv_win_kf, plv_win_cfir, 1000, srate, DELAY
)
plt.show()

envcor_kf = env_cor(gt_states.copy(), kf_delayed.copy())
envcor_cfir = env_cor(gt_states.copy(), cfir_delayed.copy())
print(
    "KF total PLV = ",
    round(np.abs(plv_tot_kf), 2),
    "CFIR total PLV = ",
    round(np.abs(plv_tot_cfir), 2),
    end=" ",
)
print("KF envcor = ", round(envcor_kf, 2), "CFIR envcor = ", round(envcor_cfir, 2))

# Plot envelope cross-correlations and delays

t, corrs_cfir = crosscorr(np.abs(gt_states), np.abs(cfir_states), srate, 200)
t, corrs_kf = crosscorr(np.abs(gt_states), np.abs(kf_states), srate, 200)
t_ms = t * 1000
plot_crosscorrelations(t_ms, corrs_cfir, corrs_kf)
plt.show()


# In[12]:


# np.savez(f"simulated_data_ar_{NOISE_AR_ORDER}.npz", gt_states=gt_states, noise_sim=noise_sim, noise_real=noise_real, sim_noise_sigma=SIM_NOISE_SIGMA_GT, real_noise_sigma=REAL_NOISE_SIGMA_GT, sr=SRATE, noise_ar_order=NOISE_AR_ORDER, freq=FREQ_GT, A=A_GT, alpha=ALPHA)
