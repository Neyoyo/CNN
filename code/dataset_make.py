#coding:utf-8
import numpy as np
from sklearn.cross_validation import train_test_split

def get_sample_data():
	input_dataset = np.load('../classify/all_23656_inputs.npy')
	input_dataset.astype(np.float32)
	np.random.shuffle(input_dataset)
	return input_dataset
def print_data(input_dataset):
	train_size = 0.80#80%作为训练集
	test_size = 1 - train_size#20%作为测试集
	input_dataset = get_sample_data()
	train, test = train_test_split(input_dataset, train_size=train_size)
	print('Compare Data Set Size')
	print('==========================')
	print('Original Dataset size: {}'.format(input_dataset.shape))
	print('Train size: {}'.format(train.shape))
	print('Test size: {}'.format(test.shape))
	return train, test, test_size
def get_class_distribution(y):
	d = {}
	set_y = set(y)
	for y_label in set_y:
		no_elements = len(np.where(y == y_label)[0])
		d[y_label] = no_elements
	dist_percentage = {class_label: count/(1.0*sum(d.values())) for class_label, count in d.items()}
	return dist_percentage
def print_class_label_split(train, test):
	#打印训练集类别分布
	y_train = train[:,-1]
	train_distribution = get_class_distribution(y_train)
	print('\n Train data set class label distribution')
	for k, v in train_distribution.items():
		print('Class label = %d, percentage records = %.2f)'%(k, v))
	
	#print测试集类别分布
	y_test = test[:,-1]
	test_distribution = get_class_distribution(y_test)
	print('\n Test data set class label distribution')
	for k, v in test_distribution.items():
		print('Class label = %d, percentage records = %.2f)'%(k, v))
	
if __name__ == '__main__':
	#train, test = print_data(input_dataset=get_sample_data())
	#print_class_label_split(train, test)
	from sklearn.cross_validation import StratifiedShuffleSplit#导入库

	input_dataset = get_sample_data()
	train, test, test_size = print_data(input_dataset)
	print_class_label_split(train, test)
	stratified_split = StratifiedShuffleSplit(input_dataset[:,-1],test_size=test_size)
	#调用sklearn里的StratifiedShuffleSplit函数。第一个参数是输入的数据集；第二个参数test_size定义了测试集的大小；第三个参数n_iter定义了只进行一次分割。
	
	for train_indx,test_indx in stratified_split:
		train = input_dataset[train_indx]
		test = input_dataset[test_indx]	
	print_class_label_split(train, test)





