import os
import configparser
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 1000

config = configparser.ConfigParser()
config.read('config.ini')

image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'filename': tf.io.FixedLenFeature([], tf.string)
}


img_size = config.getint('TRAIN','IMG_SIZE')
print(img_size)


def _parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    data = tf.io.decode_png(parsed_features['image'], 3)
    data = tf.image.resize(data, [img_size]*2, method='nearest')
    data = tf.cast(data, tf.float32)
    data = data / 255
    label = parsed_features["label"]
    return data, label

def make_batches(dataset, batch_size=32):
    """
    Generates batches from a given dataset
    """
    ds_batches = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER_SIZE))
    ds_batches = ds_batches.batch(batch_size)
    ds_batches = ds_batches.prefetch(buffer_size=AUTOTUNE)
    return ds_batches


def make_datasets(record_dir, batch_size):
    records = os.listdir(record_dir)
    train_records = [x for x in records if "train" in x]
    test_records = [x for x in records if "test" in x]
    
    train_dataset = tf.data.TFRecordDataset([os.path.join(record_dir, r) for r in train_records])
    train = train_dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    
    val_dataset = tf.data.TFRecordDataset([os.path.join(record_dir, r) for r in test_records])
    validation = val_dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    
    train_batches = make_batches(train, batch_size)
    validation_batches = make_batches(validation, batch_size)
    return train_batches, validation_batches
