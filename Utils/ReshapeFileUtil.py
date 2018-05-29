# -*-coding:utf-8-*-
import numpy as np

def ReshapeFile(path):
    data = np.loadtxt(path)
    data=data.reshape(-1)
    np.savetxt(path, data)

if __name__=='__main__':
    ReshapeFile("/home/dmrf/桌面/未命名文件夹/H_Q.txt")
