#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import dataset_make
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc  
from keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
plt.switch_backend('agg') 
		#定义参数
conv1_filter =7
conv2_filter =5
#dataset
input_dataset = dataset_make.get_sample_data()
train,test,test_size = dataset_make.print_data(input_dataset)
stratified_split = StratifiedShuffleSplit(input_dataset[:,-1],test_size=test_size)
for train_index,test_index in stratified_split:
	train = input_dataset[train_index]
	test = input_dataset[test_index]
dataset_make.print_class_label_split(train, test)
x_train = train[:,:-1]
y_train = train[:,-1]
y_train = y_train.astype('int64')
y_train = to_categorical(y_train)
x_test = test[:,:-1]
y_test = test[:,-1]
y_test = y_test.astype('int64')
y_test = to_categorical(y_test)
#extract the weight after training
pos = np.load('../classify/pos_12469_input.npy')
pos_inp = pos[:,:-1]
pos_label = pos[:,-1]
pos_label = to_categorical(pos_label)
#制作训练batch
dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size =23656)
dataset = dataset.batch(128)
iterator = dataset.make_initializable_iterator()
X,Y = iterator.get_next()
#定义权重，偏置，卷积，池化函数。
def weight_variable(shape,layer_name):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape,stddev=0.1)
        weights = tf.Variable(initial,name = 'w')
        tf.summary.histogram(layer_name+'/weights',weights)
    return weights

def biases_variable(shape,layer_name):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1,shape=shape)
        biases = tf.Variable(initial,name = 'b')
        tf.summary.histogram(layer_name+'biases',biases)
    return biases

def conv2d(x,W,layer_name):
    outputs = tf.nn.conv2d(x,W,strides =[1,1,1,1],padding = 'VALID')
    tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs

def max_pool2x1(x,layer_name):
    outputs = tf.nn.avg_pool(x,ksize = [1,2,1,1],strides = [1,2,1,1],padding='VALID')
    tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs

#制作卷集神经网络
#输入层：inputs
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None,404],name = 'xs')
    ys = tf.placeholder(tf.float32, [None,2],name = 'ys')
    keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
x_reshaped = tf.reshape(xs,[-1,101,4,1],name = 'x_reshape')
#卷积层1：
with tf.name_scope('conv1'):
    W_conv1 = weight_variable([conv1_filter,4,1,16],'W_conv1')
    b_conv1 = biases_variable([16],'b_conv1')
    conv1 = conv2d(x_reshaped,W_conv1,'conv1')
    h_conv1 = tf.nn.relu(conv1+b_conv1)
#池化层1：
with tf.name_scope('pool1'):
    h_pool1 = max_pool2x1(h_conv1,'pool1')
#卷积层2：
with tf.name_scope('conv2'):
    W_conv2 = weight_variable([conv2_filter,1,16,32],'W_conv2')
    b_conv2 = biases_variable([32],'b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,'conv2')+b_conv2)
#池化层2：
with tf.name_scope('pool2'):
    h_pool2 = max_pool2x1(h_conv2,'pool2')
h_pool2_flat = tf.reshape(h_pool2,[-1,((102-conv1_filter)//2-conv2_filter+1)//2*1*32])#拍平
#全连接层1：
with tf.name_scope('fc1'):
    W_fc1 = weight_variable([((102-conv1_filter)//2-conv2_filter+1)//2*1*32,128],'fc1')
    b_fc1 = biases_variable([128],'fc1')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#输出层即全连接2：
with tf.name_scope('fc2'):
    W_fc2 = weight_variable([128,2],'fc2')
    b_fc2 = biases_variable([2],'fc2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2,name = 'prob')
#计算loss:
with tf.name_scope('cross_entropy'):
    #loss = tf.losses.mean_squared_error(prediction,ys)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    #cross_entropy = tf.reduce_sum(tf.square(prediction - ys)) 
    tf.summary.scalar('cross_entropy',cross_entropy)
#进行训练
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
#results
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#保存模型
saver = tf.train.Saver()
tf.add_to_collection("predict", prediction)
max_acc = 0

#启动会话
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('log',sess.graph)
    sess.run(iterator.initializer)
    for batch in range(18500): 
        batch_xs,batch_ys = sess.run([X,Y])
        _,acc=sess.run([train_step,accuracy],feed_dict ={xs:batch_xs,ys:batch_ys,keep_prob:0.7})
        if batch % 185 == 0: 
            epoch = batch/185
            w_conv1,acc,pre = sess.run([W_conv1,accuracy,prediction],feed_dict ={xs:x_test,ys:y_test,keep_prob:1.0})
            print('In epoch %d;the acc is %.4f'%(epoch,acc))
            if acc>max_acc:
                max_acc = acc
                saver.save(sess,'ckpt_11_14_19_30/model.ckpt')
            rs = sess.run(merged,feed_dict={xs:x_test,ys:y_test,keep_prob:1.0})
            writer.add_summary(rs,batch)            
    ##compute roc
    #model_file = tf.train.latest_checkpoint('ckpt_11_14_19_30/')
    #saver.restore(sess,model_file)
    #weights_matrix,conv1_matrix = sess.run([W_conv1,conv1],feed_dict ={xs:pos_inp,ys:pos_label,keep_prob:1.0})
    ##save filter
    #np.save('weights1_3.npy',weights_matrix)
    #np.save('conv1_3.npy',conv1_matrix)
    #pre = pre[:,0]
    #pre = np.argmax(pre,axis=1)
    #y_test_c = y_test[:,1]
    ##calculate precision,recall,f1_score
    #p = precision_score(y_test_c,pre)
    #r = recall_score(y_test_c,pre)
    #f1 = f1_score(y_test_c,pre)
    ##calculate auc
    #plt.boxplot(pre,showfliers = False)
    #fpr, tpr, thresholds = metrics.roc_curve(y_test_c,pre)
    #roc_auc = auc(fpr,tpr)
    #plt.figure()
    #lw = 2
    #plt.figure(figsize=(10,10))
    #plt.plot(fpr, tpr, color='darkorange',         
    #	lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    #plt.show()
    #iplt.savefig('classification_pr.png')
    #print('precise is %.4f;recall_is %.4f;f1 is %.4f.'%(p,r,f1))
    print('finish')
