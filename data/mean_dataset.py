import csv

import pandas as pd

df = pd.read_csv('ovarian_cancer/results/ovarian_cancer.csv')
# raman_type = df['raman_type'].unique()
# print(raman_type)

raman_shift = df.iloc[0:1, 2:].values.reshape(1480)

spectrum_df = df[df['labels'] == 0]
spectrum_df = spectrum_df.sample(frac=1.0)
spectrum_df = spectrum_df.iloc[0:, 2:]

y = spectrum_df.mean().values.reshape(1480)
# y = spectrum_df.values.reshape(1193)


# Average spectral data were provided for grad-cam
with open("../tools/visualizations/data/resnet50/ovarian.csv", "w", encoding="utf-8", newline="") as file:
    csv_writer = csv.writer(file)
    header = ["raman_type", "labels"]
    raman_head = ["raman_shift", "NaN"]
    for i in range(len(raman_shift)):
        head_str = "Var" + str(i)
        header.append(head_str)
        raman_head.append(raman_shift[i])
    csv_writer.writerow(header)
    csv_writer.writerow(raman_head)

    spectrum = ["melanoma cells", "0"]
    for i in range(len(y)):
        spectrum_str = str(y[i])
        spectrum.append(spectrum_str)
    csv_writer.writerow(spectrum)

    file.close()
