import csv
import random

import numpy as np
import pandas as pd

path = './ovarian_cancer/results/ovarian_cancer.csv'
data = pd.read_csv(path)
raman_type = data.iloc[1:, 0:1].values
labels = data.iloc[1:, 1:2].values
spectrum = data.iloc[1:, 2:].values
print(spectrum.shape)
print(labels.shape)

# The standard deviation of Gaussian white noise for each Raman shift
length_label = len(labels)
indices = list(range(length_label))
random.shuffle(indices)


num_new_data = 100
new_spectrum = []
new_labels = []
new_type = []
for i in range(num_new_data):
    if i >= length_label:
        k = indices[i - length_label]
    else:
        k = indices[i]
    original_spectrum = spectrum[k]
    noise_std = 0.03 * np.std(original_spectrum)
    print(noise_std)
    noise = np.random.normal(scale=noise_std, size=original_spectrum.shape)
    new_data = original_spectrum + noise
    new_spectrum.append(new_data)
    new_labels.append(labels[k])
    new_type.append(raman_type[k])

new_spectrum = np.array(new_spectrum)
new_labels = np.array(new_labels)
new_type = np.array(new_type)
# Output a new data set
print(new_spectrum.shape)
print(new_labels.shape)

# 1. open the file
with open("ovarian_cancer/augument.csv", "a", encoding="utf-8", newline="") as file:
    # 2. Built on file objects csvWrite object
    csv_writer = csv.writer(file)
    # 3. Build list header
    header = ["raman_type", "labels"]
    for i in range(new_spectrum.shape[1]):
        head_str = "Var" + str(i)
        header.append(head_str)
    csv_writer.writerow(header)

    # 4. add spectrum
    for i in range(num_new_data):
        spectrum_head = [new_type[i][0], new_labels[i][0]]
        for j in range(len(new_spectrum[i])):
            spectrum_head.append(new_spectrum[i][j])

        csv_writer.writerow(spectrum_head)

    file.close()
