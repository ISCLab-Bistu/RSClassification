# Copyright (c) OpenMMLab. All rights reserved.
import os
import re

import numpy as np


# Function to parse our files
def parse_text(file, dir):
    with open(dir + file, 'rt') as fd:
        data = []
        line = fd.readline()
        nline = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        data.append(nline)
        while line:
            line = fd.readline()
            nline = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            data.append(nline)
    return data


# Function to load the data
def load_txt(file_path):
    # 
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    base_dir = "../../" + file_path + "/"
    file_list = os.listdir(base_dir)
    num_base = len(file_list)  # 

    # Start extraction
    X = []  # actual y of spectra
    Y = []  # 1 -> no_graphene; 0 -> classes
    coord = []  # actual x of spectra
    classes = file_list  # （）

    for i in range(num_base):
        # type（）
        type_dir = base_dir + file_list[i] + '/'
        all_files_type = os.listdir(type_dir)

        for f in all_files_type:
            data = []
            datab = []
            for e in parse_text(f, type_dir):
                if len(e) > 0:
                    datab.append(float(e[0]))
                    data.append(float(e[1]))
            coord.append(datab)
            X.append(data)
            Y.append(i)

    # Transform into np.array
    X = np.array(X)
    Y = np.array(Y)

    # Remove the negative values from spectra
    for i in range(len(X)):
        for j in range(len(X[i])):
            if (X[i][j] < 0):
                X[i][j] = 0

    return Y, classes, coord, X
