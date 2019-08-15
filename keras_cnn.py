from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

path = "./mnist.npz"
f = np.load(path)
X_train, y_train = f['x_train'],f['y_train']
X_test, y_test = f['x_test'],f['y_test']

class LeNet:
	def build(input_shape, classes):
		model = Sequential()
		# CONV -> RELU -> POOL
		model.add(Conv2D(20, kernel_size = 5, padding = "same", input_shape = input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
		# CONV -> RELU -> POOL
		model.add(Conv2D(50, kernel_size = 5, padding = "same", input_shape = input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model


NB_EPOCH = 4
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
K.set_image_dim_ordering("th")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = LeNet.build(input_shape = INPUT_SHAPE, classes = NB_CLASSES)
model.summary()
# print(model)
model.compile(loss = "categorical_crossentropy", optimizer = OPTIMIZER, metrics = ['accuracy'])
history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE, validation_split = VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose = VERBOSE)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()