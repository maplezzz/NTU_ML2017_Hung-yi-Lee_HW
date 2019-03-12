from utility import clean_data, plot_training_history

import pandas as pd
import numpy as np
from time import time
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Model

from sklearn.utils.class_weight import compute_class_weight

name = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

train_data = clean_data('data/train.csv')
test_data = clean_data('data/test.csv', False)

train = train_data.feature.reshape((-1, 48, 48, 1))/255
train_x = train[:-2000]
train_label = train_data.label[:-2000]
train_onehot = train_data.onehot[:-2000]
test_x = train[-2000:]
test_label = train_data.label[-2000:]
test_onehot = train_data.onehot[-2000:]


class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(train_data.label),
                                    y=train_data.label)


#CNN model

inputs = Input(shape=(48,48,1))

# First convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=64, padding='same',
             activation='relu', name='layer_conv1')(inputs)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = BatchNormalization(axis = -1)(net)
net = Dropout(0.25)(net)

# Second convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=128, padding='same',
             activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = BatchNormalization(axis = -1)(net)
net = Dropout(0.25)(net)

# Third convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=256, padding='same',
             activation='relu', name='layer_conv3')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = BatchNormalization(axis = -1)(net)
net = Dropout(0.5)(net)

# Flatten the output of the conv-layer from 4-dim to 2-dim.
net = Flatten()(net)

# First fully-connected / dense layer with ReLU-activation.
net = Dense(128)(net)
net = BatchNormalization(axis = -1)(net)
net = Activation('relu')(net)

# Last fully-connected / dense layer with softmax-activation
# so it can be used for classification.
net = Dense(7)(net)
net = BatchNormalization(axis = -1)(net)
net = Activation('softmax')(net)
# Output of the Neural Network.
outputs = net


y = model.fit(x=train_x,
           y=train_onehot,
           validation_data=(test_x, test_onehot),
           class_weight=class_weight,
           epochs=100, batch_size=64,
           callbacks=[tensorboard]
             )

plot_training_history(y)
#
# model.save('cnn.h5')

#DNN model

inputs = Input(shape=(48,48,1))

dnn = Flatten()(inputs)

dnn = Dense(512)(dnn)
dnn = BatchNormalization(axis = -1)(dnn)
dnn = Activation('relu')(dnn)
dnn = Dropout(0.25)(dnn)

dnn = Dense(1024)(dnn)
dnn = BatchNormalization(axis = -1)(dnn)
dnn = Activation('relu')(dnn)
dnn = Dropout(0.5)(dnn)

dnn = Dense(512)(dnn)
dnn = BatchNormalization(axis = -1)(dnn)
dnn = Activation('relu')(dnn)
dnn = Dropout(0.5)(dnn)

dnn = Dense(7)(dnn)
dnn = BatchNormalization(axis = -1)(dnn)
dnn = Activation('softmax')(dnn)

outputs = dnn

model2 = Model(inputs=inputs, outputs=outputs)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model2.compile(optimizer='Adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

d = model2.fit(x=train_x,
           y=train_onehot,
           validation_data=(test_x, test_onehot),
           class_weight=class_weight,
           epochs=100, batch_size=64,
           callbacks=[tensorboard]
             )

plot_training_history(d)
#model2.save('dnn.h5')