
""" linear model """
from matplotlib import pyplot as plt

from PTE import pTE

X_linear = []
Y_linear = []
Z_linear = []
f = open("doc/linear01.txt", 'r')
line = f.readline().split()
while line:
    # print(line)
    X_linear.append(float(line[0]))
    Y_linear.append(float(line[1]))
    Z_linear.append(float(line[2]))
    line = f.readline().split()
f.close()

time_series_linear = [X_linear, Y_linear, Z_linear]
causality_matrix_linear, surrogate_causality_matrix_linear = pTE(time_series_linear, tau=1, dimEmb=1, surr='iaaft', Nsurr=100)

print(f'causality_matrix = \n{causality_matrix_linear}\n')
print(f'surrogate causality matrix = \n{surrogate_causality_matrix_linear}\n')
labels = ['X', 'Y', 'Z']

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(causality_matrix_linear, interpolation='nearest', cmap=plt.cm.get_cmap('PuBu'))
fig.colorbar(cax)

ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)

plt.show()
"""
    tau: lag of the embedding
    dimEmb: dimension of the embedding (model order)
    surr: type of surrogates
    Nsurr: number of surrogates
"""