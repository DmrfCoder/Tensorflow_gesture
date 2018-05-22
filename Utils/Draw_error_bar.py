# -*- coding: utf-8 -*-

import tablib
from pyecharts import Bar
import numpy as np

ds3 = tablib.Dataset()
ds3.headers = ['name', 'error_count']
labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J']

error_numpy = np.loadtxt('../Data/abij_error_index.txt')
g = np.zeros((8), dtype=np.int64)

for a in error_numpy:
    g[int(a)] = g[int(a)]+1

for i in range(0, 8):
    ds3.append([labels[i], g[i]])
    print labels[i], g[i]

bar3 = Bar('error_count')
bar3.add('error_count', ds3.get_col(0), ds3.get_col(1))
bar3.render('../Data/abij_error_bar.html')
