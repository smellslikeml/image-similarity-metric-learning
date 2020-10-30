# Image similarity with Metric Learning

![top most similar example](https://smellslike.ml/img/top_most_similar.jpg)

This is the code referenced in the Content-Based Video Recommendations with Metric Learning [blog](https://smellslike.ml/posts/content-based-video-recommendations-with-metric-learning/). We provide a basic custom and ResNet50 based architecture for training a CNN to generate image embeddings. 

Approximate nearest neighbors for embeddings are found using the [Annoy](https://github.com/spotify/annoy) library.

## Requirements
* [TensorFlow==2.3.0](https://www.tensorflow.org/install)
* [TensorFlow add-ons](https://www.tensorflow.org/addons)
* [Annoy](https://github.com/spotify/annoy)
* Pandas
* matplotlib
* pillow

Install using the ```requirements.txt``` file:

```bash
pip install -r requirements.txt
```

## Data

Use the ```utils/create_tfrecords.py``` script to generate tfrecords from training data. Start by editing the ```config.ini``` file with the paths to your training and testing csv files. 

```
[DATA]
TRAIN_DF = /path/to/train.csv 
TEST_DF = /path/to/test.csv
TFRECORD_DIR = /path/to/datasets/image_tfrecords/
DATA_DIR = /path/to/imgs/
```

The training and testing csv files should have the following structure:
```bash
          img  action     sports  adventure  aerobics  ...  western  world history
0    0001.jpg       0          0          0         0  ...        0              0
1    0002.jpg       0          0          0         0  ...        0              0
2    0003.jpg       1          0          0         0  ...        0              0
3    0004.jpg       1          0          0         0  ...        0              0
4    0005.jpg       0          0          1         0  ...        0              0
```

A column for images and boolean columns for each genre category.

Run:
```bash
$ python utils/create_tfrecords.py
```

## Train

You can choose which model architecture to train using the ```ARCH``` variable in the config file. Simply run:

```bash
$ python train.py
```

Checkpoint files will be automatically stored in the ```checkpoints/``` directory as well as tensorboard logs in the ```logs/``` directory.

## Evaluate

Generate image embeddings using the ```utils/generate_embeddings.py``` script. Update the config file to point to a csv file containing the image paths and genre information for the samples you wish to generate embeddings.

Here is an example of the csv structure:
```bash
          img                genres
0    0001.jpg              "Action"          
1    0002.jpg            "Thriller"       
2    0003.jpg       "Action,Comedy"    
3    0004.jpg      "Comedy,Romance" 
4    0005.jpg              "Horror"          
```

In the ```config.ini``` file, update these variables:
```
[EVALUATE]
DATA_FILE = /path/to/imgs.csv
IMAGE_DIR = /path/to/imgs/
```

Then run:
```bash
$ python utils/generate_embeddings.py
```

You can generate a top 10 most similar images example using the ```utils/evaluate.py``` script.

```bash
$ python utils/evaluate.py
```
