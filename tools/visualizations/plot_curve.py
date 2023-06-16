import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import rmsm.datasets.rampy as rp

matplotlib.use('TkAgg')

data = pd.read_csv('data/pnas_cvb.csv')
x = data.iloc[0:1, 2:].values.reshape(1044)
y = data.iloc[1:2, 2:].values.reshape(1044)
x = x[88:350]
y = y[88:350]

# smooth
y_smooth = rp.smooth(x, y, method="savgol", window_length=5, polyorder=2)

# baseline
roi = np.array([[-29, 4096]])
ycalc_poly, base_poly = rp.baseline(x, y_smooth, roi, method='arPLS', lam=10 ** 6, ratio=0.001)

# normalization
y_norm = rp.normalise(y=ycalc_poly, x=x, method="minmax")

plt.grid()
plt.plot(x, y_norm)
# plt.savefig('./data/raman.jpg')

plt.show()
