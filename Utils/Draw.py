#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def Draw():
    list_acc=np.loadtxt('../Data/list_acc.txt')
    list_acc_bat=np.loadtxt('../Data/list_acc_bat.txt')
    a=[]
    b=[]
    d=0
    for i in range(0,47):
        if i%2==0:
            a.append(list_acc[i])
            b.append(d*128)
            d+=1
    print(list_acc)
    print(list_acc_bat)
    plt.plot(b,a)
    plt.savefig('../Data/acc_lstm.png',format='png')
    plt.show()


if __name__=='__main__':
    Draw()