# -*-coding:utf-8-*-
import numpy as np
import os

mic_time = 550
mic_num = 2


def mic(path, pathnew):
    files1 = os.listdir(path)
    count = 1
    for file1 in files1:
        print('处理第' + str(count))
        count = count + 1
        files2 = os.listdir(path + '/' + file1)
        a = np.loadtxt(path + '/' + file1 + '/' + files2[0])
        b = np.loadtxt(path + '/' + file1 + '/' + files2[1])
        if len(a)!=8800:
            print('error')
            continue
        if len(b)!=8800:
            print('error')
            continue
        a = a.reshape(8, 1100)
        b = b.reshape(8, 1100)



        for i in range(0, mic_num):
            try:
                os.mkdir(pathnew + '/' + file1 + '_' + str(i))
            except OSError:
                break
            c = np.arange(mic_time * 8, dtype=np.float64).reshape(8, mic_time)
            d = np.arange(mic_time * 8, dtype=np.float64).reshape(8, mic_time)
            for j in range(0, 8):
                c[j] = a[j][i * mic_time:(i + 1) * mic_time]
                d[j] = b[j][i * mic_time:(i + 1) * mic_time]

            c=c.reshape(mic_time*8)
            d=d.reshape(mic_time*8)

            if file1[-15]=='A':
                np.savetxt(pathnew + '/' + file1 + '_' + str(i)+'/'+str(0) + '_' + files2[0], c)
                np.savetxt(pathnew + '/' + file1 + '_' +str(i)+'/'+ str(0) + '_' + files2[1], d)
            if file1[-15]=='B':
                np.savetxt(pathnew + '/' + file1 + '_' + str(i)+'/'+str(1) + '_' + files2[0], c)
                np.savetxt(pathnew + '/' + file1 + '_' +str(i)+'/'+ str(1) + '_' + files2[1], d)

            if file1[-15]=='C':
                np.savetxt(pathnew + '/' + file1 + '_' + str(i)+'/'+str(2) + '_' + files2[0], c)
                np.savetxt(pathnew + '/' + file1 + '_' +str(i)+'/'+ str(2) + '_' + files2[1], d)
            if file1[-15]=='F':
                if i==0:
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(7) + '_' + files2[0], c)
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(7) + '_' + files2[1], d)
                else:
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(8) + '_' + files2[0], c)
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(8) + '_' + files2[1], d)
            if file1[-15] == 'G':
                if i == 0:
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(3) + '_' + files2[0], c)
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(3) + '_' + files2[1], d)
                else:
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(4) + '_' + files2[0], c)
                    np.savetxt(pathnew + '/' + file1 + '_' + str(i) + '/' + str(4) + '_' + files2[1], d)

            if file1[-15]=='I':
                np.savetxt(pathnew + '/' + file1 + '_' + str(i)+'/'+str(5) + '_' + files2[0], c)
                np.savetxt(pathnew + '/' + file1 + '_' +str(i)+'/'+ str(5) + '_' + files2[1], d)
            if file1[-15]=='J':
                np.savetxt(pathnew + '/' + file1 + '_' + str(i)+'/'+str(6) + '_' + files2[0], c)
                np.savetxt(pathnew + '/' + file1 + '_' +str(i)+'/'+ str(6) + '_' + files2[1], d)

if __name__ == '__main__':
    mic('/home/dmrf/下载/demodata',
        '/home/dmrf/下载/demodatanew')
