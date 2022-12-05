
""" nonlinear model """
import time

import numpy as np
from matplotlib import pyplot as plt

from PTE import pTE

def ar_model_dhamala(N=500, Trials = 10, Fs = 200, C=0.2, t_start=0, t_stop=None, cov = None):
    '''
        AR model from Dhamala et. al.
    '''

    T = N / Fs

    time = np.linspace(0, T, N)

    X = np.random.random([Trials, N])
    Y = np.random.random([Trials, N])

    def interval(t, t_start, t_stop):
        if t_stop==None:
            return ( t>=t_start)
        else:
            return ( t>=t_start ) *( t<=t_stop)

    for i in range(Trials):
        E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))
        for t in range(2, N):
            X[i ,t] = 0.55 *X[i , t -1] - 0.8 *X[i , t -2] + interval(time[t] ,t_start ,t_stop ) * C *Y[i , t -1] + E[t ,0]
            Y[i ,t] = 0.55 *Y[i , t -1] - 0.8 *Y[i , t -2] + E[t ,1]

    Z = np.zeros([Trials,2 ,N])

    Z[: ,0 ,:] = X
    Z[: ,1 ,:] = Y

    return Z


N = 5000  # Number of observations
Fs = 200  # Sampling frequency
dt = 1.0 / Fs  # Time resolution
C = 0.25  # Coupling parameter
Trials = 10  # Number of trials
freqs = np.arange(1, 100, .1)  # Frequency axis
# Covariance matrix
cov = np.array([[2.00, 0.00],
                [0.00, 1.00]])

# Generating data
X = ar_model_dhamala(N=N, Trials=Trials, C=C, Fs=Fs, t_start=0, t_stop=None, cov=cov)
x = X[0][0][:]
X_nonlinear = X[9][0][:]
Y_nonlinear = X[9][1][:]

# f = open("doc/nonlinear01.txt", 'r')
# line = f.readline().split()
# while line:
#     # print(line)
#     X_nonlinear.append(float(line[0]))
#     Y_nonlinear.append(float(line[1]))
#     line = f.readline().split()
# f.close()

time1 = time.time()
time_series_linear = [X_nonlinear, Y_nonlinear]
causality_matrix_linear, surrogate_causality_matrix_linear = pTE(time_series_linear, tau=1, dimEmb=1, surr='iaaft', Nsurr=100)

print(f'causality_matrix = \n{causality_matrix_linear}\n')
print(f'surrogate causality matrix = \n{surrogate_causality_matrix_linear}\n')

labels = ['X', 'Y']

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(causality_matrix_linear, interpolation='nearest', cmap=plt.cm.get_cmap('PuBu'))
fig.colorbar(cax)

ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)
time2 = time.time()
time3 = time2-time1

plt.plot([25,500],[0.05,time3])
plt.xlabel("length")
plt.ylabel("time")

# print(time3)
plt.show()
"""
    tau: lag of the embedding
    dimEmb: dimension of the embedding (model order)
    surr: type of surrogates
    Nsurr: number of surrogates
"""