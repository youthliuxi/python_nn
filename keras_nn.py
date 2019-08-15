from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
np.random.seed(1671)

# 网络和训练
NB_EPOCH = 20
BATCH_SIZE = 120
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
# 训练中用作验证的数据集比例为20%
DROPOUT = 0.3

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
path = "./mnist.npz"
f = np.load(path)
X_train, y_train = f['x_train'],f['y_train']
X_test, y_test = f['x_test'],f['y_test']
RESHAPED = 784
NB_HIDDEN = 128

X_train = X_train.reshape(60000,RESHAPED)
X_test = X_test.reshape(10000,RESHAPED)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# print(X_test)

# 归一化
X_train /= 255
X_test /= 255
print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
print(Y_test.shape)
print(type(X_test))

model = Sequential()
# model.add(Dense(NB_CLASSES, input_shape = (RESHAPED,)))
# model.add(Activation('softmax'))

# 加一个隐藏层
model.add(Dense(NB_HIDDEN, input_shape = (RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))


model.summary()
model.compile(loss = 'categorical_crossentropy',optimizer = OPTIMIZER, metrics = ['accuracy'])

history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE, validation_split = VALIDATION_SPLIT)
# history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE)

score = model.evaluate(X_test, Y_test, verbose = VERBOSE)
print("Test score:", score[0])
print("Test accuracy", score[1])