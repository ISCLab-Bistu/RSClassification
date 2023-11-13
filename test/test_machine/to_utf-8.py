# Bulk change txt to utf-8
# Note that the.py file needs to be run in the same folder as the txt file

import os
from os import path
from chardet import detect


def to_utf8(path):
    fileSuffix = 'csv'
    fns = []
    filedir = os.path.join(os.path.abspath(path), "")
    file_name = os.listdir(os.path.join(os.path.abspath(path), ""))
    for fn in file_name:
        if fn.endswith(fileSuffix):
            fns.append(os.path.join(filedir, fn))
    for fn in fns:
        with open(fn, 'rb+') as fp:
            content = fp.read()
            if len(content) == 0:
                continue
            else:
                codeType = detect(content)['encoding']
                content = content.decode(codeType, "ignore").encode("utf8")
                fp.seek(0)
                fp.write(content)
                print(fn, "：Changed to utf8 encoding")


to_utf8('data/')
