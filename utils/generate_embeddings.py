import configparser
import pandas as pd
import tensorflow as tf
import tensorflow_addons

# Importing variables
config = configparser.ConfigParser()
config.read('config.ini')

model = tf.keras.models.load_model(config.get('EVALUATE', 'MODEL_FILE'))

class TestImageGenerator(tf.keras.utils.Sequence) :
    def __init__(self, image_filenames, img_size, batch_size):
        self.image_filenames = image_filenames
        self.img_size = img_size
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

        imgs = np.array([image.img_to_array(image.load_img(i,target_size=(self.img_size, self.img_size, 3)))/255 for i in batch_x])
        return imgs

# Created file for all embeddings
all_filenames_array = np.array(data['img'])
imgGenerator = TestImageGenerator(all_filenames_array, batch_size)

for idx in range(imgGenerator.__len__()):
    img_batch = imgGenerator.__getitem__(idx)
    embeddings = new_model.predict(img_batch).tolist()
    with open(save_embedding_path, "a") as outfile:
        for emb in embeddings:
            outfile.write(";" + str(emb) + "\n")


emb_df = pd.read_csv(save_embedding_path, names=["embedding"], delimiter=";")
emb_df = emb_df.reset_index()
emb_df = emb_df.drop(["index"], axis=1)

emb_df["img"] = data["img"]
emb_df = emb_df[["img", "embedding"]]

all_images = data["img"].tolist()
movie_labels = []
for pic in all_images:
    ex = data.loc[data["img"] == pic].drop(["img"], axis=1)
    ex_genres = ex.columns[(ex == 1).all()].tolist()
    movie_labels.append(",".join(ex_genres))

emb_df["genres"] = movie_labels

emb_df.to_csv(save_embedding_path, index=False)

