# conding=utf-8
import argparse
import sys
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
	mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
	x = tf.compat.v1.placeholder(tf.float32, [None,784])
	y_ = tf.compat.v1.placeholder(tf.float32, [None,10])

	# deepnn
	x_image = tf.reshape(x, [-1,28,28,1])
	print(x_image)
	W_conv1 = weight_variable([5,5,1,32])
	print(W_conv1)
	b_conv1 = bias_variable([32])
	print(b_conv1)
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	print(h_conv1)

	h_pool1 = max_pool_2x2(h_conv1)
	print(h_pool1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	print(h_conv2)

	h_pool2 = max_pool_2x2(h_conv2)
	print(h_pool2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	print(h_fc1)

	keep_prob = tf.compat.v1.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, rate = 1 - keep_prob)
	print(h_fc1_drop)

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	print(y_conv)

	# 以上是神经网络层
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = y_conv)
	print(cross_entropy)
	cross_entropy = tf.reduce_mean(cross_entropy)
	print(cross_entropy)
	train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	print(correct_prediction)
	correct_prediction = tf.cast(correct_prediction,tf.float32)
	print(correct_prediction)
	accuracy = tf.reduce_mean(correct_prediction)
	print(accuracy)
	
	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.global_variables_initializer())
		for i in range(20000):
			batch = mnist.train.next_batch(100)
			if i % 1000 == 0:
				train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
				print(str(i) + '\\' + str(train_accuracy))

			train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})
		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
	initial = tf.random.truncated_normal(shape, stddev=0.1)
	# 正态分布，均值为0，方差为0.1
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	# 常量0.1
	return tf.Variable(initial)
if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# # 创建一个可解析对象
	# parser.add_argument()
	# # 向该对象添加你要关注的命令行参数和选项
	# parser.parse_args()
	# # 进行解析
	# # 以上涉及命令行解析问题
	old_v = tf.compat.v1.logging.get_verbosity()
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	main()
	tf.compat.v1.logging.set_verbosity(old_v)