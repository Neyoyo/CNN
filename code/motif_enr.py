import numpy as np
pos = np.load('conv1_4.npy')
reference = np.load('/home/lfliu/TSS_data/classify/pos_12469_input.npy')
reference = reference[:,:-1]
w = np.load('weights1_4.npy')
reference = reference.reshape([12469,101,4])
pos = pos.reshape([12469,95,16])
p_value = [np.sum(np.max(w[:,:,:,i],axis = 1))*0.6 for i in range(16)]
for n in range(16):
    filter1 = pos[:,:,n]
    index = np.arange(95)
    max_value = [max(filter1[i,filter1[i,:]>p_value[n]],default = 0) for i in range(filter1.shape[0])]
    instance = [index[filter1[i,:] == max_value[i]] for i in range(filter1.shape[0])]
    motif = [reference[i,j:j+7,:] for i in range(filter1.shape[0]) for j in list(instance[i])]
    #instance_all = [index[filter1[i,:]>p_value[n]] for i in range(filter1.shape[0])]
    #motif_all = [reference[i,j:j+7,:] for i in range(filter1.shape[0]) for j in list(instance_all[i])]
    m = n+1
    np.save('./instance9/filter_06_%d_instance.npy'% m,motif)
    print(np.array(motif).shape)
