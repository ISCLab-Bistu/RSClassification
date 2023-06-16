import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

import rmsm.datasets.rampy as rp

mpl.use('TkAgg')


def plot_curve(raman_shift, spectrum):
    x = raman_shift
    y = spectrum
    y_norm = y
    # y[y < 0] = 0

    # smooth
    y_smooth = rp.smooth(x, y, method="whittaker", Lambda=10 ** 0.5)

    # baseline
    roi = np.array([[-29, 4090]])
    ycalc_poly, base_poly = rp.baseline(x, y_smooth, roi, 'arPLS', lam=10 ** 5, p=0.05)

    # normalization
    y_norm = rp.normalise(y=ycalc_poly, x=x, method="minmax")

    return x, y_norm


# plot raman spectrum for dataset
df = pd.read_csv('./single_cell/results/single_cell.csv')

start = 71
end = 425
raman_shift = df.iloc[0:1, start + 2:end + 2].values
raman_shift = raman_shift.flatten()
# print(raman_shift)

df1 = df[df['labels'] == 1]

labels = ['labels']
df1 = df1.drop(labels, axis=1)

spectrum1 = df1.iloc[1:, start + 1:end + 1]
spectrum1 = spectrum1.sample(frac=1.0).values
# spectrum1 = spectrum1[0, :]
spectrum1 = spectrum1.mean(axis=0)

x_normal, y_normal1 = plot_curve(raman_shift, spectrum1)
# with open("health.txt", mode="w", encoding='utf-8') as file:
#     for i in range(len(y_normal1)):
#         save_str = str(y_normal1[i])
#         save_str = save_str[1:(len(save_str)-1)]
#         write_f = save_str + "\n"
#         file.write(write_f)
#
# with open("T12-Tis.txt", mode="w", encoding='utf-8') as file:
#     for i in range(len(y_chazhiTis)):
#         save_str = str(y_chazhiTis[i])
#         save_str = save_str[1:(len(save_str) - 1)]
#         write_f = save_str + "\n"
#         file.write(write_f)
#


plt.plot(x_normal, y_normal1, label='T1', color='blue')

plt.xlabel("Raman shift cm$^{-1}$", fontsize=12)
plt.ylabel("Normalized intensity", fontsize=12)
# plt.ylim(-0.5, 5.5)
# plt.yticks([0, 1, 2, 3, 4, 5])
# plt.grid()
# plt.legend()
plt.show()
