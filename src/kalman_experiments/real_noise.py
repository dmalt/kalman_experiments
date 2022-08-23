import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw_brainvision
from scipy.signal import welch

matplotlib.use("TkAgg")

# raw_path = "/home/altukhov/Data/real_eyes/Koleno.vhdr"


class RealNoise:
    def __init__(self, single_channel_eeg: np.ndarray):
        self.single_channel_eeg = single_channel_eeg
        self.ind = 0

    def step(self) -> float:
        n_samp = len(self.single_channel_eeg)
        if self.ind >= len(self.single_channel_eeg):
            raise IndexError(f"Index {self.ind} is out of bounds for data of length {n_samp}")
        self.ind += 1
        return self.single_channel_eeg[self.ind]


def prepare_real_noise(
    raw_path: str, sigma: float = 1, minsamp: int = 0, maxsamp: int | None = None
) -> RealNoise:
    raw = read_raw_brainvision(raw_path, preload=True)
    raw.pick_channels(["FC2"])
    raw.crop(tmax=244)
    raw.filter(l_freq=0.1, h_freq=None)

    data = np.squeeze(raw.get_data())
    data /= data.std()
    data -= data.mean()
    print("dm=", data.mean())
    data *= sigma
    crop = slice(minsamp, maxsamp)
    return RealNoise(data[crop])


if __name__ == "__main__":
    o = 1
    alpha = 1
    # an = ArNoise(np.array([0] * o), alpha=1, order=o, sigma=199)
    y0 = np.random.rand(o)
    print(y0.shape)
    an = prepare_real_noise(
        raw_path="ds004148/sub-01/ses-session2/eeg/sub-01_ses-session2_task-eyesopen_eeg.vhdr"
    )
    res = []
    for i in range(100_000):
        res.append(an.step())
    res = np.array(res)
    # import matplotlib.pyplot as plt
    # from scipy.signal import welch
    freqs, psd = welch(res, fs=500, nperseg=1024)
    plt.plot(freqs[1:-1], psd[1:-1])
    plt.plot(freqs[1:-1], [1 / f**alpha for f in freqs[1:-1]])
    plt.yscale("log")
    plt.xscale("log")
    plt.legend([f"AR({o}) for 1/f noise", "1/f"])
    plt.grid()
    plt.show()
    # plt.plot(res)

    plt.show()

# freqs, psd = welch(data, fs=raw.info["sfreq"], nperseg=1024)

# alpha = 1
# freqs_lim = slice(1, None)
# plt.plot(freqs[freqs_lim], psd[freqs_lim])
# plt.plot(freqs[freqs_lim], [1 / f**alpha for f in freqs[freqs_lim]])
# plt.plot(freqs[freqs_lim], [1 / f**1.1 for f in freqs[freqs_lim]])
# plt.plot(freqs[freqs_lim], [1 / f**1.5 for f in freqs[freqs_lim]])
# plt.yscale("log")
# plt.xscale("log")
# plt.legend(["Real noise", "1/f", "$1/f^{1.5}$", r"$1/f^{1.1}$"])
# plt.show()
