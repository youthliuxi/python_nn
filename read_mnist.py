# # 第一种mnist数据的读取方式
# # 从网上下载mnist数据库读取
# # t10k-images-idx3-ubyte.gz
# # t10k-labels-idx1-ubyte.gz
# # train-images-idx3-ubyte.gz
# # train-labels-idx1-ubyte.gz
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# import tensorflow as tf

# old_v = tf.compat.v1.logging.get_verbosity()
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(len(mnist.train.images))
# print(len(mnist.train.labels))
# print(len(mnist.test.images))
# print(len(mnist.test.labels))
# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)
# print(mnist.test.images.shape)
# print(mnist.test.labels.shape)
# tf.compat.v1.logging.set_verbosity(old_v)

# # 第二种mnist数据的读取方式
# # 从csv文件读取
# training_data_file = open("mnist_dataset/mnist_train.csv",'r')
# training_data_list = training_data_file.readlines()
# training_data_file.close()
# for record in training_data_list:
# 	all_values = record.split(',')
# 	inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
# 	targets = numpy.zeros(output_nodes) + 0.01
# 	targets[int(all_values[0])] = 0.99
# 	n.train(inputs,targets)
# 	pass
# test_data_file = open("mnist_dataset/mnist_test.csv",'r')
# test_data_list = test_data_file.readlines()
# test_data_file.close()
# scorecard = []
# for record in test_data_list:
# 	all_values = record.split(',')
# 	correct_label = int(all_values[0])
# 	# print(correct_label,"correct_label")
# 	inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
# 	outputs = n.query(inputs)
# 	label = numpy.argmax(outputs)
# 	# print(label,"network's answer")

# 	if (label == correct_label):
# 		scorecard.append(1)
# 	else:
# 		scorecard.append(0)
# 		pass
# 	pass

# 第三种mnist数据的读取方式
# 从mnist.npz文件读取
import numpy as np
path = "./mnist.npz"
f = np.load(path)
X_train, y_train = f['x_train'],f['y_train']
X_test, y_test = f['x_test'],f['y_test']
print(f['x_train'].shape)
print(f['y_train'].shape)
print(f['x_test'].shape)
print(f['y_test'].shape)