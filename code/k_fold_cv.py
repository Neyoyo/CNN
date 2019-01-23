#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import dataset_make
from keras.utils import to_categorical
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
plt.switch_backend('agg')   
#定义参数
acc_pre_re_spe_gm_f1_score = []
pre_sum = []
y_test_sum = []
for i in range(3,6,2):
    for j in range(3,5,2):

        conv1_filter =i
        conv2_filter =j
        #dataset
        input_dataset = dataset_make.get_sample_data()
        X = input_dataset[:,:-1]
        Y = input_dataset[:,-1]
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        for train,test in kfold.split(X,Y):
            y_train = Y[train].astype('int64')
            y_train = to_categorical(y_train)
            y_test = Y[test].astype('int64')
            y_test = to_categorical(y_test)
            x_train = X[train]
            x_test = X[test]
            dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size =18924)
            dataset = dataset.batch(128)
            iterator = dataset.make_initializable_iterator()
            Xs,Ys = iterator.get_next()
            #定义权重，偏置，卷积，池化函数。
            def weight_variable(shape,layer_name):
                with tf.name_scope('weights'):
                    initial = tf.truncated_normal(shape,stddev=0.1)
                    weights = tf.Variable(initial,name = 'w')
                    #tf.summary.histogram(layer_name+'/weights',weights)
                return weights
            
            def biases_variable(shape,layer_name):
                with tf.name_scope('biases'):
                    initial = tf.constant(0.1,shape=shape)
                    biases = tf.Variable(initial,name = 'b')
                    #tf.summary.histogram(layer_name+'biases',biases)
                return biases
            
            def conv2d(x,W,layer_name):
                outputs = tf.nn.conv2d(x,W,strides =[1,1,1,1],padding = 'VALID')
               # tf.summary.histogram(layer_name+'/outputs',outputs)
                return outputs
            
            def max_pool2x1(x,layer_name):
                outputs = tf.nn.avg_pool(x,ksize = [1,2,1,1],strides = [1,2,1,1],padding='VALID')
                #tf.summary.histogram(layer_name+'/outputs',outputs)
                return outputs
            
            #制作卷集神经网络
            #输入层：inputs
            with tf.name_scope('inputs'):
                xs = tf.placeholder(tf.float32, [None,404],name = 'input')
                ys = tf.placeholder(tf.float32, [None,2],name = 'output')
                keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
            x_reshaped = tf.reshape(xs,[-1,101,4,1],name = 'x_reshape')
            #卷积层1：
            with tf.name_scope('conv1'):
                W_conv1 = weight_variable([conv1_filter,4,1,16],'conv1')
                b_conv1 = biases_variable([16],'conv1')
                h_conv1 = tf.nn.relu(conv2d(x_reshaped,W_conv1,'conv1')+b_conv1)
            #池化层1：
            with tf.name_scope('pool1'):
                h_pool1 = max_pool2x1(h_conv1,'pool1')
            #卷积层2：
            with tf.name_scope('conv2'):
                W_conv2 = weight_variable([conv2_filter,1,16,32],'conv2')
                b_conv2 = biases_variable([32],'conv2')
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
                prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
            #计算loss:
            with tf.name_scope('cross_entropy'):
                #loss = tf.losses.mean_squared_error(prediction,ys)
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
                #cross_entropy = tf.reduce_sum(tf.square(prediction - ys)) 
                #tf.summary.scalar('cross_entropy',cross_entropy)
            #进行训练
            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
            #results
            correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            saver = tf.train.Saver()
            max_acc = 0 
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                #merged = tf.summary.merge_all()
                #writer = tf.summary.FileWriter('log1',sess.graph)
                sess.run(iterator.initializer)
                for batch in range(15000): 
                    batch_xs,batch_ys = sess.run([Xs,Ys])
                    _,acc=sess.run([train_step,accuracy],feed_dict ={xs:batch_xs,ys:batch_ys,keep_prob:0.7})
                    if batch % 147 == 0: 
                        epoch = batch/147
                        w_conv1,acc,pre = sess.run([W_conv1,accuracy,prediction],feed_dict ={xs:x_test,ys:y_test,keep_prob:1.0})
                        if acc>max_acc:
                            max_acc = acc
                            saver.save(sess,'ckpt3/model.ckpt')
                        #rs = sess.run(merged,feed_dict={xs:x_test,ys:y_test,keep_prob:1.0})
                        #writer.add_summary(rs,batch)   
                model_file = tf.train.latest_checkpoint('ckpt3/') 
                saver.restore(sess,model_file)
                val_acc,pre = sess.run([accuracy,prediction],feed_dict ={xs:x_test,ys:y_test,keep_prob:1.0})
                pre_c = pre[:,1]
                pre = np.argmax(pre,axis = 1)
                y_test_c = y_test[:,1]
                #calculate tp,fp,tn,fn
                tn,fp,fn,tp = confusion_matrix(y_test_c,pre).ravel()
                precision = precision_score(y_test_c,pre)
                recall = recall_score(y_test_c,pre)
                specificity = tn/(tn+fp)
                gm = np.sqrt(recall*specificity)
                f1 = f1_score(y_test_c,pre)
                #calcualte auc
                fpr, tpr, thresholds = metrics.roc_curve(y_test_c,pre_c)
                roc_auc = auc(fpr,tpr)
                #summary all the results into a list
                acc_pre_re_spe_gm_f1_score.append(val_acc)
                acc_pre_re_spe_gm_f1_score.append(precision)
                acc_pre_re_spe_gm_f1_score.append(recall)
                acc_pre_re_spe_gm_f1_score.append(specificity)
                acc_pre_re_spe_gm_f1_score.append(gm)
                acc_pre_re_spe_gm_f1_score.append(f1)
                acc_pre_re_spe_gm_f1_score.append(roc_auc)
            tf.reset_default_graph()
            print('1')
a = np.array(acc_pre_re_spe_gm_f1_score)  
np.save('cvscores.npy',a)  

##calculate the acc in dataset
#final_pre = np.hstack((b[i] for i in range(5)))
#y_actual = np.hstack((c[i] for i in range(5)))
#tn,fp,fn,tp = confusion_matrix(y_actual,final_pre).ravel()
#precision = precision_score(y_actual,final_pre)
#recall = recall_score(y_actual,final_pre)
#specificity = tn/(tn+fp)
#gm = np.sqrt(recall*specificity)
#f1 = f1_score(y_actual,final_pre)
#

        
       
