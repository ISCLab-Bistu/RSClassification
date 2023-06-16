import csv
import random

import numpy as np
import pandas as pd

path = './cell_spectrum/results/cell_spectrum.csv'
data = pd.read_csv(path)
raman_type = data.iloc[1:, 0:1].values
labels = data.iloc[1:, 1:2].values
spectrum = data.iloc[1:, 2:].values
print(spectrum.shape)
print(labels.shape)

length_label = len(labels)
indices = list(range(length_label))
random.shuffle(indices)


num_new_data = 350
new_spectrum = []
new_labels = []
new_type = []
for i in range(num_new_data):
    if i >= length_label:
        k = indices[i - length_label]
    else:
        k = indices[i]
    original_spectrum = spectrum[k]
    intensity_factor = np.random.uniform(0.2, 2)
    new_data = original_spectrum * intensity_factor
    new_spectrum.append(new_data)
    new_labels.append(labels[k])
    new_type.append(raman_type[k])

new_spectrum = np.array(new_spectrum)
new_labels = np.array(new_labels)
new_type = np.array(new_type)


with open("augument.csv", "a", encoding="utf-8", newline="") as file:
    csv_writer = csv.writer(file)
    # Header
    header = ["raman_type", "labels"]
    for i in range(len(labels)):
        head_str = "Var" + str(i)
        header.append(head_str)
    csv_writer.writerow(header)

    # 4. Spectrum
    for i in range(num_new_data):
        spectrum_head = [new_type[i][0], new_labels[i][0]]
        for j in range(len(new_spectrum[i])):
            spectrum_head.append(new_spectrum[i][j])

        csv_writer.writerow(spectrum_head)

    file.close()
