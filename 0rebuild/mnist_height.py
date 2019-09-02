import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D
from keras.layers import MaxPool2D,Flatten,Dropout,ZeroPadding2D,BatchNormalization
from keras.utils import np_utils
import keras
from keras.models import save_model,load_model
from keras.models import Model
from keras.datasets import mnist

batch_size = 64
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

n_filters = 32
pool_size = (2,2)

#设定相关参数

cnn_net = Sequential()

cnn_net.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1)))
cnn_net.add(Activation('relu'))
cnn_net.add(MaxPool2D(pool_size=pool_size))

cnn_net.add(ZeroPadding2D((1,1)))
cnn_net.add(Conv2D(48,kernel_size=(3,3)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size=pool_size))

cnn_net.add(ZeroPadding2D((1,1)))
cnn_net.add(Conv2D(64,kernel_size=(2,2)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size = pool_size))

cnn_net.add(Dropout(0.25))
cnn_net.add(Flatten())

cnn_net.add(Dense(3168))
cnn_net.add(Activation('relu'))

cnn_net.add(Dense(10))
cnn_net.add(Activation('softmax'))
#查看网络结构

cnn_net.summary()

from keras.utils.vis_utils import plot_model,model_to_dot
from IPython.display import Image,SVG

SVG(model_to_dot(cnn_net).create(prog='dot',format='svg'))
#模型训练

cnn_net.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

cnn_net.fit(x_train,y_train,batch_size=batch_size,epochs=50,verbose=1,
             validation_split=0.2)
cnn_net.save('cnn_net_model_eg1.h5')
#上述过程在我的个人PC电脑上跑了4个小时，我都崩溃了，
#我把训练好的模型保存了下来，下次使用直接加载
Model = load_model('cnn_net_model_eg1.h5')
score = Model.evaluate(x_test, y_test, batch_size=128)
print(str(Model.metrics_names[0])+':'+str(score[0])+'\n'+
          str(Model.metrics_names[1])+':'+str(score[1]))
pred = Model.predict(x_test,batch_size=128)
#这个预测是softmax给的说以是概率，职能自己手动转化成label
test_pred = np.argmax(pred,axis=1)
y_test1 = np.argmax(y_test,axis=1)
#plot
import matplotlib.pyplot as plt
# %matplotlib inline
actuals = y_test1[0:6]
predictions = test_pred[0:6]
images = np.squeeze(x_test[0:6])
Nrows = 2
Ncols = 3
for i in range(6):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(images[i], [28,28]), cmap='Greys_r')
    plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
              fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)