import os
import pandas as pd
import tensorflow as tf
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

image_feature_description = {
    'label': tf.io.FixedLenFeature([1], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'filename': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    data = tf.io.decode_jpeg(parsed_features['image'], 3)
    label = parsed_features["label"]
    filename = parsed_features["filename"]
    return data, label, filename

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    try:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    except:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def image_example(filename, label, flip=False): 
    image_string = open(filename, 'rb').read()
    image_array = tf.image.decode_jpeg(image_string)
    filename = filename.split("/")[-1]
    feature = {
            'label': _int64_feature(label),
            'image': _bytes_feature(image_string),
            'filename': _bytes_feature(bytes(filename, 'utf-8')),
            }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def process_df(df, image_col, data_dir):
    """
    Takes pandas dataframe and returns list of 
    image paths and list of labels
    """
    images = df[image_col].tolist()
    images = [os.path.join(data_dir, img) for img in images]
    labels = df.drop([image_col], axis=1).values.tolist()
    return images, labels

def create_records(df, image_col, data_dir, output_dir, num_of_records=10, prefix="posters_"):
    """
    Takes a pandas dataframe and number of records to create and creates TFRecords.
    Saves records in output_dir
    """
    total_records = 0
    images, labels = process_df(df, image_col, data_dir)

    record_prefix = os.path.join(output_dir, prefix)
    files_per_record = int(len(images) / num_of_records)  #approximate number of images per record 
    chunk_number = 0

    for i in range(0, len(images), files_per_record):
        print("Writing chunk ", str(chunk_number))
        images_chunk = images[i:i+files_per_record]
        labels_chunk = labels[i:i+files_per_record]

        record_file = record_prefix + str(chunk_number).zfill(3) + ".tfrecords"

        with tf.io.TFRecordWriter(record_file) as writer:
            for idx, image in enumerate(images_chunk):
                sample_labels = np.where(np.array(labels_chunk[idx]) != 0)[0]
                for l in sample_labels:
                    tf_example = image_example(image, l)
                    writer.write(tf_example.SerializeToString())
                    total_records += 1
            chunk_number += 1
    return total_records

if __name__ == "__main__":
    train_df = pd.read_csv(config['DATA']['TRAIN_DF'])
    test_df = pd.read_csv(config['DATA']['TRAIN_DF'])
    output_dir = config['DATA']['TFRECORD_DIR']
    data_dir = config['DATA']['DATA_DIR']
    image_col = config['DATA']['IMG_COL']


    num_of_train_records = 20
    num_of_test_records = 10
    create_recs = True

    if create_recs:
        train_record_count = create_records(train_df, image_col, data_dir, output_dir, num_of_train_records, prefix="posters_train-")
        test_record_count = create_records(test_df, image_col, data_dir, output_dir, num_of_test_records, prefix="posters_test-")

        print("####   TOTAL RECORDS WRITTEN  ######")
        print("TRAIN: ", train_record_count)
        print("TEST: ", test_record_count)
    else:
        records = os.listdir(output_dir)
        train_records = [x for x in records if "train" in x]
        test_records = [x for x in records if "test" in x]

        train_dataset = tf.data.TFRecordDataset([os.path.join(output_dir, r) for r in train_records])
        train = train_dataset.map(_parse_function)

        limit = 5

        for image, label, filename in train.take(limit):
            print(image.numpy(), filename.numpy(), label.numpy())

