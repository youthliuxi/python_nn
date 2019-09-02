# -*- coding:utf-8 -*-
import os
import matplotlib
import json
import re
import matplotlib.pyplot as plt

def drawPlt(data,jpgName):
	data_str = ['val_loss','val_acc','loss','acc']
	plt.figure(figsize=(10,8))
	for i in range(0,4):
		plt.subplot(221+i)
		plt.plot(data[data_str[i]])
		plt.title(data_str[i])
	# plt.show()
	plt.savefig(".\\%s.png" % (jpgName))
	print(".\\%s.png" % (jpgName))
	plt.close()

def main():
	path = ".\\mnist_2.txt"

	with open(path, 'r', encoding = 'utf-8') as f:
		# print(f)
		file_txt = f.readline()
		file_txt = re.sub('\'','\"',file_txt)
		# 将单引号替换双引号
		data =json.loads(file_txt)
		# print(data['val_loss'])
		drawPlt(data,"mnist_2")

		

if __name__ == '__main__':
	main()