import os
import os.path as osp
from os import PathLike


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


data = expanduser('~/.config/')
print(data)
