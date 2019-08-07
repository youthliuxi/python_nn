import numpy
import matplotlib.pyplot as plt
import pylab
import scipy.special
import time
start_time = time.time()

class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		self.lr = learningrate
		# 权重随机矩阵
		# self.wih = (numpy.random.rand(self.hnodes, self.inodes)-0.5)
		# self.who = (numpy.random.rand(self.onodes, self.hnodes)-0.5)
		# 权重矩阵，标准正态分布
		# 参数分别是：中心值，标准差，numpy数组大小
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		# 激活函数 sigmoid函数
		self.activating_function = lambda x:scipy.special.expit(x)
		pass

	def train(self, inputs_list, targets_list):
		inputs = numpy.array(inputs_list, ndmin = 2).T
		targets = numpy.array(targets_list, ndmin = 2).T
		
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activating_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activating_function(final_inputs)
		
		output_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.who.T, output_errors)
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
		
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))

		pass

	def query(self, inputs_list):
		inputs = numpy.array(inputs_list, ndmin = 2).T
		# .T 转置
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activating_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activating_function(final_inputs)
		return final_outputs

input_nodes = 784
hidden_nodes = 1000
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# 将数据处理成28*28的矩阵
# all_values = data_list[0].split(',')
# image_array = numpy.asfarray(all_values[1:]).reshape([28,28])
# [1:]用来忽略第一个值，该值为标签
# print(image_array)
# plt.imshow(image_array, cmap = 'Greys',interpolation = 'None')
# plt.show()
# 有plt.show()才会显示图片
epochs = 2
for e in range(epochs):
	for record in training_data_list:
		all_values = record.split(',')
		inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
		targets = numpy.zeros(output_nodes) + 0.01
		targets[int(all_values[0])] = 0.99
		n.train(inputs,targets)
		pass
	pass
train_time = time.time()
test_data_file = open("mnist_dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# all_values = test_data_list[0].split(',')
# image_array = numpy.asfarray(all_values[1:]).reshape([28,28])
# plt.imshow(image_array, cmap = 'Greys',interpolation = 'None')
# # plt.show()
# print(n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01))

scorecard = []
for record in test_data_list:
	all_values = record.split(',')
	correct_label = int(all_values[0])
	# print(correct_label,"correct_label")
	inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
	outputs = n.query(inputs)
	label = numpy.argmax(outputs)
	# print(label,"network's answer")

	if (label == correct_label):
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
	pass
# print(scorecard)
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
end_time = time.time()
print("train_time:", train_time - start_time)
print("test_time:", end_time - train_time)
print("all_time:", end_time - start_time)
# scaled_input = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
# print(scaled_input)

# onodes = 10
# targets = numpy.zeros(onodes) + 0.01
# targets[int(all_values[0])] = 0.99
# print(target)