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

def draw_mj():
    labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J']
    y_true = np.loadtxt('../Data/re_label_lstm.txt')
    y_pred = np.loadtxt('../Data/pr_label_lstm.txt')

    a=np.zeros((10),dtype=np.int64)
    b=np.zeros((10),dtype=np.int64)

    for i in range(0,1000):
        print(str(i))
        if y_true[i]>9:
            continue
        if y_true[i] <0:
            continue

        if y_pred[i]>9:
            continue
        if y_pred[i] <0:
            continue

        a[int(y_true[i])]+=1
        if(y_true[i]==y_pred[i]):
            b[int(y_true[i])] += 1

    for i in range(0,10):
        if a[i]!=0:
            print(str(i)+':'+str(b[i]/a[i]))


def draw_iq():
    i=np.loadtxt('/home/dmrf/文档/Gesture/New_Data/持续时间为1s的复杂手势/连续手势集/Train/B_3124428/B_I_3124428.txt')
    q=np.loadtxt('/home/dmrf/文档/Gesture/New_Data/持续时间为1s的复杂手势/连续手势集/Train/B_3124428/B_Q_3124428.txt')
    i=i.reshape(-1)
    q=q.reshape(-1)
    plt.plot(i,q)
    plt.show()

if __name__=='__main__':
   # Draw()
    #draw_mj()
    draw_iq()