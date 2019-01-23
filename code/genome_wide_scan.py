import tensorflow as tf
import numpy as np
import os
import math
import time
import sys
import re
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session()
new_saver=tf.train.import_meta_graph('ckpt_11_14_19_30/model.ckpt.meta')
new_saver.restore(sess,"ckpt_11_14_19_30/model.ckpt")
graph = tf.get_default_graph()
x=graph.get_operation_by_name('inputs/xs').outputs[0]
keep_prob = graph.get_operation_by_name('inputs/keep_prob').outputs[0]
y = graph.get_operation_by_name('fc2/prob').outputs[0]
sess.close()
def genome_wide_scan(line,index_priro,chromosome_id,out_dir):
    chromosome = 'chr'+str(chromosome_id)
    out=open(out_dir,'w+')
    for index,base in enumerate(line):
        if index<=(len(line)-101):
            start = index;end = index+101
            lin = line[start:end]
            lin = lin.upper()
            lin = lin.replace('A','1 0 0 0 ')
            lin = lin.replace('C','0 1 0 0 ')
            lin = lin.replace('G','0 0 1 0 ')
            lin = lin.replace('T','0 0 0 1 ')
            lin = lin.replace('N','0 0 0 0 ')
            lin = lin[:-1]
            lin = lin.strip()
            lin = lin.split(' ')
            start = index+int(index_priro);end = index+int(index_priro)+101 
            sample = np.array(lin)
            sample = sample.astype(np.float32)
            sample = sample.reshape([1,404])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                prediction = sess.run(y,feed_dict={x:sample,keep_prob:1})
                result = np.argmax(prediction,1)
                out.write('%s\t%s\t%s\t%s\n' % (chromosome,start,end,result))
                out.flush()

genome = sys.argv[1]
index_pri = sys.argv[2]
chrom_id=sys.argv[3]
out_file = sys.argv[4]
f = open(genome,'r')
input_sequence=''
for line in f:
    if re.match('>',line) is None:
        input_sequence=input_sequence+line.strip()
f.close()
chr_sequence=input_sequence
#start = time.time()
genome_wide_scan(line=chr_sequence,index_priro=index_pri,chromosome_id=chrom_id,out_dir=out_file)
#end = time.time()
#print(end-start)

