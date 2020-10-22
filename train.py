import os
import datetime
import numpy as np
from utils.datasets import make_datasets
import configparser
from model import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa

# Importing variables
config = configparser.ConfigParser()
config.read('config.ini')

# GPU strategy
strategy = tf.distribute.MirroredStrategy()

# Checkpoint formatting
train_time = datetime.datetime.now().strftime("%d_%m_%Y-%H-%M-%S")
checkpoint_path = os.path.join(config['TRAIN']['CHECKPOINT_DIR'], config['TRAIN']['NAME'] + "_checkpoint_{}.h5".format(train_time))
print("Checkpointing at ", checkpoint_path)

# Initializing datasets
batch_size = config.getint('TRAIN', 'BATCH_SIZE') * strategy.num_replicas_in_sync
steps_per_epoch = round(config.getint('TRAIN','TRAIN_SIZE')/ batch_size)
train_batches, validation_batches = make_datasets(config['DATA']['TFRECORD_DIR'], batch_size=batch_size)


with strategy.scope():
    m = Model(name=config['TRAIN']['ARCH'])
    model = m.getModel()
    if config['TRAIN']['ARCH'] == 'resnet':
        for layer in model.layers[:249]:
           layer.trainable = False
        for layer in model.layers[249:]:
           layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.getfloat('TRAIN', 'LR')),
        loss=tfa.losses.TripletSemiHardLoss(distance_metric=config['TRAIN']['DISTANCE_METRIC'], margin=config.getfloat('TRAIN', 'MARGIN')))

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')
tensorboard_view = tf.keras.callbacks.TensorBoard(log_dir=config['TRAIN']['LOG_DIR'], profile_batch=2)
callbacks_list = [tensorboard_view, checkpoint]

model.fit(train_batches,
                   steps_per_epoch = steps_per_epoch,
                   epochs = config.getint('TRAIN','EPOCHS'),
                   verbose = 1,
                   validation_data = validation_batches,
                   validation_steps = config.getint('TRAIN','VAL_STEPS'),
                   callbacks=callbacks_list)

model.save(os.path.join(config['TRAIN']['MODEL_DIR'], config['TRAIN']['NAME'] + "_{}.h5".format(train_time)))
