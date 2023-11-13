import csv

import pandas as pd


base_dir = 'ovarian_cancer/'
file_name = 'ovarian_cancer.csv'
path = base_dir + file_name

data = pd.read_csv(path)

shift = data['raman_shift'].values[0]
raman_shift = shift.split(",")

spectrum = data['spectrum'].values
raman_type = data['raman_type'].values
labels = data['label'].values

# 1. Create the file object (specify filename, mode, encoding)
with open(base_dir + "sers/" + file_name, "a", encoding="utf-8", newline="") as file:
    # 2. Build the csv write object from the file object
    csv_writer = csv.writer(file)
    # 3. Header
    header = ["raman_type", "labels"]
    raman_head = ["raman_shift", "NaN"]
    for i in range(len(raman_shift)):
        head_str = "Var" + str(i)
        header.append(head_str)
        raman_head.append(raman_shift[i])
    csv_writer.writerow(header)
    csv_writer.writerow(raman_head)

    # 4. Spectrum
    for i in range(len(spectrum)):
        # 
        spectrum_head = [raman_type[i], labels[i]]
        spectrum_list = spectrum[i].split(",")
        print(len(spectrum_list))
        for j in range(len(spectrum_list)):
            spectrum_head.append(spectrum_list[j])

        csv_writer.writerow(spectrum_head)

    # 5. 
    file.close()
