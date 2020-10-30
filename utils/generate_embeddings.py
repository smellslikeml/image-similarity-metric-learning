import os
import glob
import pandas as pd
import numpy as np
import configparser
import tensorflow as tf
import tensorflow_addons
from model import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image

# Importing variables
config = configparser.ConfigParser()
config.read('config.ini')

checkpoint_model_dir = glob.glob(os.path.join(config.get("TRAIN", "CHECKPOINT_DIR"), "*"))
checkpoint_model_path = max(checkpoint_model_dir, key=os.path.getctime)

model = tf.keras.models.load_model(checkpoint_model_path)
data = pd.read_csv(config.get("EVALUATE", "DATA_FILE"))

img_size = config.getint("TRAIN", "IMG_SIZE")

class ImageEmbeddingGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, image_genres, batch_size):
        self.image_filenames = image_filenames
        self.image_genres = image_genres
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_imgs = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_genres = self.image_genres[idx * self.batch_size : (idx+1) * self.batch_size]

        imgs = np.array([image.img_to_array(image.load_img(i,target_size=(img_size,img_size,3)))/255 for i in batch_imgs])
        embeddings = model.predict(imgs)
        return batch_imgs.tolist(), embeddings.tolist(), batch_genres.tolist()

embeddings_dict = {"img": [], "embedding": [], "genres": []}

# Created file for all embeddings
all_filenames_array = np.array(data['img'])
all_genres_array = np.array(data['genres'])
imgGenerator = ImageEmbeddingGenerator(all_filenames_array, all_genres_array, config.getint("TRAIN", "BATCH_SIZE"))

for idx in range(imgGenerator.__len__()):
    imgs, embeddings, gnrs = imgGenerator.__getitem__(idx)
    embeddings_dict["img"] += imgs
    embeddings_dict["embedding"] += embeddings
    embeddings_dict["genres"] += gnrs

emb_df = pd.DataFrame(embeddings_dict)
output_dir = config.get("EVALUATE", "RESULTS")
emb_df.to_csv(os.path.join(output_dir, "embeddings_{}.csv".format(checkpoint_model_path.split("/")[-1].replace(".h5", ""))), index=False)

print(emb_df.head())
