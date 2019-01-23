#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_seq(path,kind):
    input_path = os.path.join(path,'%s_seq.npy' %kind)
    label_path = os.path.join(path,'%s_label.npy' %kind)
    with open(label_path,'rb') as lbpath:
        labels = np.load(label_path)
    with open(input_path,'rb') as inputpath:
        input_data = np.load(inputpath).reshape(len(labels),4000)
    return input_data,labels
import matplotlib.pyplot as plt
#加载数据
x_train,y_train = load_seq('/home/lfliu/cnn/input_data/',kind = 'train')
x_test,y_test = load_seq('/home/lfliu/cnn/input_data/',kind = 'test')


conv1 = 21
conv2 = 5
train_num = 25000
BATCH_SIZE = 100
input_queue = tf.train.slice_input_producer([x_train,y_train],num_epochs = 100,shuffle = True)
batch_xs,batch_ys = tf.train.shuffle_batch(input_queue,batch_size = BATCH_SIZE,capacity = 20000,min_after_dequeue = 4000,num_threads = 2)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#偏置设置函数，加小的正值来避免死节点
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#2维卷积函数：https://www.cnblogs.com/qggg/p/6832342.html；P80
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')
#最大池化函数：https://blog.csdn.net/mao_xiao_feng/article/details/53453926；P80
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,10,1,1],strides=[1,10,1,1],padding='VALID')    

#placeholder
x = tf.placeholder(tf.float32,[None,4000])
y_ = tf.placeholder(tf.float32,[None,1])
x_image = tf.reshape(x,[-1,1000,4,1])
#第一个卷积层（32个建立在x_image上的5*5，1颜色通道的卷积核加上偏置后经行最大池化）
W_conv1 = weight_variable([conv1,4,1,16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二个卷积层（64个建立在h_pool1上的5*5，32颜色通道的卷积核加上偏置后经行最大池化
W_conv2 = weight_variable([conv2,1,16,32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#全链接层将[n,7,7,64]的h_pool2先转换为[n,7*7*64]的h_pool2_flat,经relu(),转换为[1024]的h_fc1
W_fc1 = weight_variable([((1001-conv1)//10-conv2+1)//10*1*32,128])
b_fcl = bias_variable([128])
h_pool2_flat = tf.reshape(h_pool2,[-1,((1001-conv1)//10-conv2+1)//10*1*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fcl)
#Dropout层，减轻过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#将Dropout层和softmax层链接,作为最后的概率
W_fc2 = weight_variable([128,1])
b_fc2 = bias_variable([1])
y_conv= tf.matmul(h_fc1_drop,W_fc2)+b_fc2
#定义损失和优化器
mse = tf.reduce_mean(tf.square(y_conv-y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)
#定义评测准确率的操作
#训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
        while not coord.should_stop():
             xs ,ys = sess.run([batch_xs,batch_ys])
             _,loss = sess.run([train_step,mse],feed_dict={x:xs,y_:ys,keep_prob:0.5 })          
    except tf.errors.OutOfRangeError:
        print('finish')
    finally:
        coord.request_stop()
    coord.join(threads)
    prediction_value = sess.run(y_conv,feed_dict={x:x_test,y_:y_test,keep_prob:1.0})

def pearson(x,y):
    n = len(x)
    sumx = sum([float(x[i]) for i in range(n)])
    sumy = sum([float(y[i]) for i in range(n)])
    sumxSq = sum([x[i]**2.0 for i in range(n)])
    sumySq = sum([y[i]**2.0 for i in range(n)])
    pSum = sum([x[i]*y[i] for i in range(n)])
    num = pSum - (sumx*sumy/n)
    den = ((sumxSq-pow(sumx,2)/n)*(sumySq - pow(sumy,2)/n))**.5
    if den == 0 :
        return 0
    r = num/den
    print(r)

pearson(prediction_value,y_test)



