import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense


def selectModel(mode):

    model = Sequential()

    if mode == 'cnn':

        model.add(
            Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu", input_shape=(48, 48, 1),
                   kernel_initializer="random_uniform", use_bias=True, bias_initializer="zeros"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(
            Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer="random_uniform", use_bias=True, bias_initializer="zeros"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(
            Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer="random_uniform", use_bias=True, bias_initializer="zeros"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(
            Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer="random_uniform", use_bias=True, bias_initializer="zeros"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(
            Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer="random_uniform", use_bias=True, bias_initializer="zeros"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(Flatten())

        model.add(Dense(512, kernel_initializer="random_uniform", activation="relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(512, kernel_initializer="random_uniform", activation="relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(7, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
    return model


trainData = np.load("data/train.npz")
x_train = trainData['Image']
y_train = trainData['Label']
model = selectModel("cnn")
model.fit(x_train, y_train, batch_size=128, epochs=20)






