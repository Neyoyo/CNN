# -*- coding: utf-8 -*-
import numpy
numpy.set_printoptions(suppress=True)
for n in range(1,17):
    
    ins_1 = numpy.load('./instance9/filter_06_'+str(n)+'_instance.npy')
    ppm = ins_1.mean(0)
    ppm = numpy.around(ppm,decimals = 4)
    #numpy.savetxt('./pwm9/filter_' +str(n)+'.csv',ppm,delimiter = ',')
    print(ins_1.shape[0])
