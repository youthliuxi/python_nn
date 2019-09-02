from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

IMG_ROWS = 28
IMG_COLS = 28
input_shape = (IMG_ROWS, IMG_COLS, 1)
NB_CLASSES = 10
OPTIMIZER = Adam()
NB_EPOCH = 100
BATCH_SIZE = 128
VALIDATION_SPLIT = 0
VERBOSE = 1

path = "../mnist.npz"
f = np.load(path)
X_train, y_train = f['x_train'],f['y_train']
X_test, y_test = f['x_test'],f['y_test']
# print(f['x_train'].shape)
# print(f['y_train'].shape)
# print(f['x_test'].shape)
# print(f['y_test'].shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# print(X_train.shape)
# print(X_test.shape)
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

def model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding = 'valid', activation = 'relu', input_shape = input_shape))
	# model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding = 'valid', activation = 'relu'))
	# model.add(Activation('relu'))
	# model.add(BatchNormalization(epsilon=1e-6,axis=1))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(rate = 1 - 0.25))
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(rate = 1 - 0.25))
	model.add(Dense(10, activation = 'softmax'))
	# model.add(Activation('softmax'))
	return model

# print(model())
model = model()
model.summary()

model.compile(loss = 'categorical_crossentropy',optimizer = OPTIMIZER, metrics = ['accuracy'])

# hist = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE, validation_data=(X_test, Y_test), validation_split = VALIDATION_SPLIT)
hist = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose = VERBOSE)
print("Test score:", score[0])
print("Test accuracy", score[1])

# print(score)
# print(hist.history)
# log_file_name = "mnist_keras.txt"
# with open(log_file_name,'w') as f:
# 	f.write(str(hist.history) + "\n")
# 	f.write(str(score) + "\n")
# 	f.write("Test score: %s\n" % score[0])
# 	f.write("Test accuracy: %s\n" % score[1])