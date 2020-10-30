import os
import ast
import glob
import numpy as np
import pandas as pd
import configparser
import seaborn as sns
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn import random_projection
from sklearn.metrics.pairwise import cosine_similarity

from mpl_toolkits.axes_grid1 import ImageGrid

# Importing variables
config = configparser.ConfigParser()
config.read('config.ini')

# Retrieving data
embedding_df_dir = glob.glob(os.path.join(config.get("EVALUATE", "RESULTS"), "*.csv"))
embedding_df_path = max(embedding_df_dir, key=os.path.getctime)

embedding_df = pd.read_csv(embedding_df_path)
num_of_records = embedding_df.shape[0]

all_images = embedding_df["img"].values.tolist()
all_embeddings = embedding_df["embedding"].values.tolist()
all_embeddings = [ast.literal_eval(emb) for emb in all_embeddings]

all_genres = embedding_df["genres"].tolist()
all_index = list(range(num_of_records))

index_to_image = dict(zip(all_index, all_images))
index_to_embedding = dict(zip(all_index, all_embeddings))
index_to_genre = dict(zip(all_index, all_genres))

embedding_dimension = 2048
index = AnnoyIndex(embedding_dimension, "dot")

# We unbatch the dataset because Annoy accepts only scalar (id, embedding) pairs.
for idx, embedding in enumerate(all_embeddings):
    index.add_item(idx, embedding)

# Build a N-tree ANN index.
N = 100
index.build(N)
index.save(os.path.join(config.get("EVALUATE", "RESULTS"), "embeddings_index_{}.ann".format(embedding_df_path.split("/")[-1].replace(".csv", ""))))

def generateViz(mySet, name):
    image_set = []
    genre_set = []
    for mov in mySet:
        query_embedding = index_to_embedding[mov]
        candidates = index.get_nns_by_vector(query_embedding, 10)
        genre_set.append(index_to_genre[mov])
        image_set.append(index_to_image[mov])
        genre_set += [index_to_genre[i] for i in candidates]
        image_set += [index_to_image[i] for i in candidates]

    fig = plt.figure(figsize=(100,100))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
            nrows_ncols=(len(mySet), 11),  # creates number of driver videos x 11 grid of axes
            axes_pad=0.5,  # pad between axes in inch.
            )

    for ax, im in zip(grid, [Image.open(x).resize((480,640)) for x in image_set]):
        ax.imshow(im)

    for ax, label in zip(grid, genre_set):
        ax.text(0.5,-0.1, label, size=12, ha="center",
                transform=ax.transAxes)
        ax.axis('off')

    plt.axis('off')
    plt.savefig(os.path.join(config.get("EVALUATE", "RESULTS"), name))

driver_image_set = [random.choice(all_index) for _ in range(5)]
generateViz(driver_image_set, "top_10_{}.png".format(embedding_df_path.split("/")[-1].replace(".csv", ""))
