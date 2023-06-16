# -*- coding: utf-8 -*-
import pandas as pd

# data = pd.read_csv('734b.csv')
# result = data.values
# print(result)
#
# for i in range(len(result)):
#     path = 'train/classes/'
#     with open(path + "/milk_powder" + str(i) + ".txt", "w", encoding='utf-8')as f:
#         for j in range(len(result[i])):
#             f.write(str(result[0][j]) + " " + str(result[i][j]) + "\n")

import os
import numpy as np

file_list = os.listdir('val/type1/')


def delete_lines(filename, head, tail):
    fin = open(filename, 'r')
    a = fin.readlines()
    fout = open(filename, 'w')
    b = ''.join(a[head:-tail])
    fout.write(b)


for f in file_list:
    file = r'val/no_graphene/' + f
    data = np.loadtxt(file)
    length = len(data)
    print(len(data))
    if (len(data) > 1941):
        head = (int)((length - 1941) / 2)
        tail = (int)((length - 1941) - head)
        delete_lines(file, head, tail)
