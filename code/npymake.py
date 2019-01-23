import numpy as np
import os

def get_seq(filename):
    s =[]
    with open(filename,'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            i += 1
            if i % 2 ==0:
                s.append(line)
    m = []
    for i in s:
        if len(i) == 102:
            i = i.upper()
            i = i.replace('A','1 0 0 0 ')
            i = i.replace('C','0 1 0 0 ')
            i = i.replace('G','0 0 1 0 ')
            i = i.replace('T','0 0 0 1 ')
            i = i.replace('N','0 0 0 0 ')
            m.append(i[:-1])
    l = len(m)
    s = []
    for n in range(l):
        m[n] = m[n].strip()
        m[n] = m[n].split(' ')
    a = np.array(m)
    m = []
    a = a.astype(int)
    print(a.shape)
    filename = filename.split('.')
    print(filename)
    np.save(filename[0]+'.npy',a)
for i in range(1,86):
    get_seq('/home/lfliu/chr2R_split/'+ str(i)+'.txt')
