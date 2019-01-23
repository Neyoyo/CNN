# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:03:15 2018

@author: 83632
"""

import numpy as np
conv1 = np.load(r'C:\Users\83632\Desktop\conv1.npy')
conv1 = conv1.reshape([12469,95,16])

print(conv1.shape)
w = np.load(r'C:\Users\83632\Desktop\weights1.npy')
pos =np.load(r'C:\Users\83632\Desktop\pos_12469.npy')
pos = pos.reshape([12469,101,4])
print(w.shape)
print(pos.shape)
for n in range(16):
    index = np.arange(95)
    filter1 = conv1[:,:,0]
    p = [np.sum(np.max(w[:,:,:,i],axis = 1))*0.6 for i in range(16)]
    instance = [index[filter1[i,:]>p[n]]for i in range(12469)]
    print(np.array(m).shape)
    m = n+1
    motif = [pos[i,j:j+7,:] for i in range(12469) for j in list(instance[i])]
    np.save('filter%d_instance.npy'% m,motif)
    print(np.array(motif).shape)
