import os
import numpy as np

file_list = os.listdir('val/type0/')


def delete_lines(filename, head, tail):
    fin = open(filename, 'r')
    a = fin.readlines()
    fout = open(filename, 'w')
    b = ''.join(a[head:-tail])
    fout.write(b)


# file = r'normal.txt'
# delete_lines(file, 0, 1)

for f in file_list:
    file = r'val/classes/' + f
    data = np.loadtxt(file)
    print(len(data))
    # if (len(data) == 1413):
        # delete_lines(file, 0, 1)
