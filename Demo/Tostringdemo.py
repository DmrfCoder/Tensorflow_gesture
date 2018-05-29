# -*-coding:utf-8-*-
import numpy as np

data = np.ndarray((2, 3, 4), dtype=np.int64)

for i in range(0,2):
    for j in range(0,3):
        for k in range(0,4):
            data[i][j][k]=i*2+j*3+k*4


print(data)

print(data.reshape(-1))
