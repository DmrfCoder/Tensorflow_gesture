# -*-coding:utf-8-*-
import csv
import numpy as np
import os


def Normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]


path = '/home/dmrf/下载/demodata'
gesturename = 'A'

csv_path = '/home/dmrf/下载/demo.csv'

with open(csv_path) as csvfile:
    gesture_len = 550
    reader = csv.reader(csvfile)
    a = []
    I = np.arange(8)
    Q = np.arange(8)
    iindex = 0
    qindex = 0
    filenameindex = 0
    prelabelindex = 0
    count = 0

    for i, rows in enumerate(reader):
        if i == 0:
            a = rows
            for k in range(0, len(a)):
                if a[k][0] == 'I':
                    iindex = k
                if a[k][0] == 'Q':
                    qindex = k
                if a[k] == 'filename':
                    filenameindex = k
                if a[k] == 'Pre_label':
                    prelabelindex = k

            print(I)
            print(Q)
            print a
            continue
        else:
            arri = np.arange(gesture_len * 8, dtype=np.float64)
            arrq = np.arange(gesture_len * 8, dtype=np.float64)

            for j in range(0, 4):  # row[j]是对应行的j列
                if j == filenameindex:
                    continue
                if j == prelabelindex:
                    continue
                v = rows[j][1:int(len(rows[j]) - 1)]
                da = v.split(',')
                l = int(len(da))
                if l != gesture_len*8:
                    print('error_len:' + str(l))
                    break

                w = 0
                if a[j][0] == 'I':
                    for w in range(0, l):

                        arri[w] = float(da[w])
                if a[j][0] == 'Q':
                    for w in range(0, l):

                        arrq[w] = float(da[w])

            if l == gesture_len*8:
                os.mkdir(path + '/' + rows[prelabelindex] + '_' + rows[filenameindex])
                arri = Normalize(arri)
                arrq = Normalize(arrq)
                np.savetxt(path + '/' + rows[prelabelindex] + '_' + rows[filenameindex] + '/' + rows[
                    filenameindex] + '_I' + '.txt', arri)
                np.savetxt(path + '/' + rows[prelabelindex] + '_' + rows[filenameindex] + '/' + rows[
                    filenameindex] + '_Q' + '.txt', arrq)
                count += 1
                print('success:' + str(count))
