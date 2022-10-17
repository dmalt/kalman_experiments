import numpy as np


def make_pink_noise(alpha, L, dt):
    """Given an alpha value for the 1/f^alpha produce data of length L and at Fs = 1/dt"""
    x1 = np.random.randn(L)
    xf1 = np.fft.fft(x1)
    A = np.abs(xf1)
    phase = np.angle(xf1)

    df = 1 / (dt * len(x1))
    faxis = np.arange(len(x1) // 2 + 1) * df
    faxis = np.concatenate([faxis, faxis[-2:0:-1]])

    oneOverf = 1 / faxis**alpha
    oneOverf[0] = 0

    Anew = np.sqrt((A**2) * oneOverf.T)
    xf1new = Anew * np.exp(1j * phase)
    return np.real(np.fft.ifft(xf1new)).T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    dt = 0.01
    noise = make_pink_noise(alpha=1, L=1024 * 10, dt=dt)
    freqs, psd_noise = welch(noise, fs=1 / dt, nperseg=512)
    plt.loglog(freqs, psd_noise)
    plt.show()
