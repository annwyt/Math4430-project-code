import time

import numpy as np
import matplotlib.pyplot  as plt
import pygc.spectral_analysis.time_frequency as tf
import pygc.non_parametric
import pygc.granger
from tqdm import tqdm


def ar_model_dhamala(N=5000, Trials=10, Fs=200, C=0.2, t_start=0, t_stop=None, cov=None):
    '''
        AR model from Dhamala et. al.
    '''

    T = N / Fs

    time = np.linspace(0, T, N)

    X = np.random.random([Trials, N])
    Y = np.random.random([Trials, N])

    def interval(t, t_start, t_stop):
        if t_stop == None:
            return (t >= t_start)
        else:
            return (t >= t_start) * (t <= t_stop)

    for i in range(Trials):
        E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))
        for t in range(2, N):
            X[i, t] = 0.55 * X[i, t - 1] - 0.8 * X[i, t - 2] + interval(time[t], t_start, t_stop) * C * Y[i, t - 1] + E[
                t, 0]
            Y[i, t] = 0.55 * Y[i, t - 1] - 0.8 * Y[i, t - 2] + E[t, 1]

    Z = np.zeros([Trials, 2, N])

    Z[:, 0, :] = X
    Z[:, 1, :] = Y

    return Z


N = 5000  # Number of observations
Fs = 200  # Sampling frequency
dt = 1.0 / Fs  # Time resolution
C = 0.25  # Coupling parameter
Trials = 10  # Number of trials
freqs = np.arange(1, 100, .1)  # Frequency axis
# Covariance matrix
cov = np.array([[1.00, 0.00],
                [0.00, 1.00]])

# Generating data
X = ar_model_dhamala(N=N, Trials=Trials, C=C, Fs=Fs, t_start=0, t_stop=None, cov=cov)

W = tf.wavelet_transform(data=X, fs=Fs, freqs=freqs, n_cycles=freqs / 2,
                         time_bandwidth=None, delta=1, method='morlet', n_jobs=-1)

# Auto- and cross-spectra
S11 = W[:, 0, :, :] * np.conj(W[:, 0, :, :])
S22 = W[:, 1, :, :] * np.conj(W[:, 1, :, :])
S12 = W[:, 0, :, :] * np.conj(W[:, 1, :, :])
S21 = W[:, 1, :, :] * np.conj(W[:, 0, :, :])
# Spectral matrix
S = np.array([[S11, S12],
              [S21, S22]]) / N
# Average over trials
S_mu = S.mean(axis=2)

S_mu = S_mu.sum(axis=-1) * dt

plt.plot(freqs, S_mu[0,0].real, label = r'$X_{1}$')
plt.plot(freqs, S_mu[1,1].real, label = r'$X_{2}$')
plt.ylabel('Power')
plt.xlabel('Frequency [Hz]')
plt.legend()

Snew, Hnew, Znew = pygc.non_parametric.wilson_factorization(S_mu, freqs, Fs, Niterations=30, verbose=True)
Ix2y, Iy2x, Ixy = pygc.granger.granger_causality(Snew, Hnew, Znew)

# Auto- and cross-spectra
S11 = W[:, 0, :, :] * np.conj(W[:, 0, :, :])
S22 = W[:, 1, :, :] * np.conj(W[:, 1, :, :])
S12 = W[:, 0, :, :] * np.conj(W[:, 1, :, :])
S21 = W[:, 1, :, :] * np.conj(W[:, 0, :, :])
# Spectral matrix
S = np.array([[S11, S12],
              [S21, S22]]) / N
# Average over trials
S_mu = S.mean(axis=2)

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.imshow(S_mu[0, 0].real, aspect='auto', cmap='jet', origin='lower', extent=[0, N / Fs, freqs[0], freqs[-1]])
plt.title(r'Power $X_{1}$')
plt.ylabel('Frequency [Hz]')
plt.subplot(2, 1, 2)
plt.imshow(S_mu[1, 1].real, aspect='auto', cmap='jet', origin='lower', extent=[0, N / Fs, freqs[0], freqs[-1]])
plt.title(r'Power $X_{2}$')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.tight_layout()
time1 = time.time()
GC = np.zeros([2, len(freqs), S_mu.shape[-1]])
for t in tqdm(range(S_mu.shape[-1])):
    Snew, Hnew, Znew = pygc.non_parametric.wilson_factorization(S_mu[:, :, :, t], freqs, Fs, Niterations=30,
                                                                verbose=False)
    GC[0, :, t], GC[1, :, t], _ = pygc.granger.granger_causality(S_mu[:, :, :, t], Hnew, Znew)
time2 = time.time()
time3 = time2 - time1
plt.plot([25,500],[0.06,time3/20])
plt.xlabel("length")
plt.ylabel("time")

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.imshow(GC[0].real, aspect='auto', cmap='jet', origin='lower', extent=[0, N / Fs, freqs[0], freqs[-1]])
plt.title(r'$X_{1}\rightarrow X_{2}$')
plt.ylabel('Frequency [Hz]')
plt.subplot(2, 1, 2)
plt.imshow(GC[1].real, aspect='auto', cmap='jet', origin='lower', extent=[0, N / Fs, freqs[0], freqs[-1]])
plt.title(r'$X_{2}\rightarrow X_{1}$')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.tight_layout()






































