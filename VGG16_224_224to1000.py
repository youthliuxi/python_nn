from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import cifar10

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape = (3,224,224)))
#第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
#对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
model.add(Conv2D(64, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
#layer2 32*32*64
model.add(Conv2D(64, (3, 3), activation = 'rule'))
#下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
#padding默认是valid，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same'))
model.add(ZeroPadding2D((1, 1)))
#layer3 16*16*64
model.add(Conv2D(128, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
#layer4 16*16*128
model.add(Conv2D(128, (3, 3), activation = 'rule'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same'))
model.add(ZeroPadding2D((1, 1)))
#layer5 8*8*128
model.add(Conv2D(256, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
#layer6 8*8*256
model.add(Conv2D(256, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
#layer7 8*8*256
model.add(Conv2D(256, (3, 3), activation = 'rule'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same'))
model.add(ZeroPadding2D((1, 1)))
#layer8 4*4*256
model.add(Conv2D(512, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
#layer9 4*4*512
model.add(Conv2D(512, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
#layer10 4*4*512
model.add(Conv2D(512, (3, 3), activation = 'rule'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same'))
model.add(ZeroPadding2D((1, 1)))
#layer11 2*2*512
model.add(Conv2D(512, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
model.add(Dropout(0.4))
#layer12 2*2*512
model.add(Conv2D(512, (3, 3), activation = 'rule'))
model.add(ZeroPadding2D((1, 1)))
#layer13 2*2*512
model.add(Conv2D(512, (3, 3), activation = 'rule'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same'))
model.add(Flatten())
#layer14 1*1*4096
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#layer15 4096
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#layer16 1000
model.add(Dense(1000))
model.add(Activation('softmax'))
# 10
model.summary()