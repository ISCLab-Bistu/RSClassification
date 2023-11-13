import csv
import os
import re


def parse_text(file):
    with open(file, 'rt') as fd:
        data = []
        line = fd.readline()
        nline = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        data.append(nline)
        while line:
            line = fd.readline()
            nline = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            data.append(nline)
    return data


def read_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # ，
            print(file_path)
            file_result = parse_text(file_path)
            for e in parse_text(file_path):
                print(e)


'''
   Generate the specified csv file according to the Raman spectroscopy file (txt or excel file) in the folder. 
   A single folder generates a single category, and then all the categories are grouped into a csv 
'''
flag_head = True
file_name = 'Health'
folder_path = './oral_degree/results/' + file_name
with open("oral_degree/results/" + file_name + '.csv', "a", encoding="utf-8", newline="") as save_csv:
    csv_writer = csv.writer(save_csv)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            header = ["raman_type", "labels"]
            raman_shift_head = ["raman_shift", "NaN"]
            spectrum_head = [file_name, 5]
            for e in parse_text(file_path):
                if len(e) > 0:
                    # if (float(e[0]) < 0):
                    #     e[0] = 0
                    # if (float(e[1]) < 0):
                    #     e[1] = 0
                    if flag_head:
                        raman_shift_head.append(float(e[0]))
                    spectrum_head.append(float(e[1]))

            for k in range(len(raman_shift_head) - 2):
                head_str = "Var" + str(k)
                header.append(head_str)
            if flag_head:
                csv_writer.writerow(header)
                csv_writer.writerow(raman_shift_head)
                flag_head = False
            csv_writer.writerow(spectrum_head)
            print("")

save_csv.close()
