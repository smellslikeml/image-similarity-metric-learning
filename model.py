import configparser

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda

from tensorflow.keras.applications import ResNet50

# Importing variables
config = configparser.ConfigParser()
config.read('config.ini')

class Model():
    def __init__(self, name='custom'):
        self.name = name
        self.img_shape = (config.getint("TRAIN","IMG_SIZE"), config.getint("TRAIN","IMG_SIZE"), 3)
        self.arch = [(64, 11, 2, 'relu'), (64, 7, 1, 'relu'), (128, 5, 1, 'relu'), (256, 3, 1, 'relu'), (256, 1, 1, None)]

    def getModel(self):
        if self.name == 'resnet':
            model = Sequential([ResNet50(include_top=False, input_shape=self.img_shape, weights='imagenet', pooling='avg'),
                    Flatten(),
                    Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
                    ])
        elif self.name == 'custom':
            netLayers = []
            netLayers.append(Conv2D(filters=self.arch[0][0], kernel_size=self.arch[0][1], strides=self.arch[0][2], padding='same', activation=self.arch[0][3], input_shape=self.img_shape))
            netLayers.append(MaxPooling2D(pool_size=2))
            for idx in range(1, len(self.arch)):
                netLayers.append(Conv2D(filters=self.arch[idx][0], kernel_size=self.arch[idx][1], strides=self.arch[idx][2], padding='same', activation=self.arch[idx][3]))
                netLayers.append(MaxPooling2D(pool_size=2))
            netLayers.append(Flatten())
            netLayers.append(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

            model = Sequential(netLayers)
        return model
