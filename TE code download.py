import time

from matplotlib import pyplot as plt

from copent import transent
from pandas import read_csv
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
prsa2010 = read_csv(url)
# index: 5(PM2.5),6(Dew Point),7(Temperature),8(Pressure),10(Cumulative Wind Speed)
data = prsa2010.iloc[2200:2700,[5,8]].values
time1 = time.time()
te = np.zeros(64)
for lag in range(1,65):
	te[lag-1] = transent(data[:,0],data[:,1],lag)
	str = "TE from pressure to PM2.5 at %d hours lag : %f" %(lag,te[lag-1])
	print(str)
te = te.reshape(8,8)
time2 = time.time()
time3 = time2-time1
# labels = ['X', 'Y']
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(te, interpolation='nearest', cmap=plt.cm.get_cmap('PuBu'))
# fig.colorbar(cax)
#
# ax.set_xticklabels(['']+labels)
# ax.set_yticklabels(['']+labels)
plt.plot([25,500],[0.06,time3])

plt.xlabel("length")
plt.ylabel("time")

# print(time3)

plt.show()