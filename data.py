import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import keras
import collections
import random
from tqdm import tqdm
import re
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
import pickle
import string
from time import time
from keras.preprocessing import sequence
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

inceptionv3 = InceptionV3(weights="imagenet", include_top=False)
modelCNN = Model(inceptionv3.input, inceptionv3.layers[-1].output)


def preprocess_img_id(img_id):
  img_folder = './content/drive/My Drive/datasets/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/'
  image_path = img_folder + img_id
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels = 3)
  img = tf.image.resize(img, (299,299))
  img = preprocess_input(img)
  return img, image_path


def preprocess_img_path(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = preprocess_input(img)
    return img, image_path


class data_loader:
    def __init__(
        self, features_shape, attention_features_shape, batch_size, buffer_size, top_k
    ):
        self.features_shape = features_shape
        self.attention_features_shape = attention_features_shape
        self.img_folder = "./content/drive/My Drive/datasets/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/"
        self.img_folder_np = "./content/drive/My Drive/datasets/Flickr8k/Flickr8k_Dataset/Flicker8k_numpy/"
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.top_k = top_k
        #
        self.train_id, self.test_id = self.load_id("train"), self.load_id("test")
        self.train_path, self.test_path = (
            self.load_path(self.train_id),
            self.load_path(self.test_id),
        )
        self.train_captions, self.test_captions = (
            self.load_caption(self.train_id),
            self.load_caption(self.test_id),
        )
        self.tokenizing()
        self.cap_train, self.name_train = self.create_split(self.train_captions)
        self.cap_test, self.name_test = self.create_split(self.test_captions)


    
    def save_npy(self):
        encode = sorted(set(self.train_id + self.test_id))
        image_dataset = tf.data.Dataset.from_tensor_slices(encode)
        image_dataset = image_dataset.map(preprocess_img_id, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(32)
        for img, path in tqdm(image_dataset):
            batch_features = modelCNN(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0],-1, batch_features.shape[3]))
            for bf, p in zip(batch_features, path):
                path_of_features = p.numpy().decode('utf-8')+'.npy'
                path_of_features = path_of_features.split('/')
                path_of_features[-2] = 'Flicker8k_numpy'
                path_of_features = '/'.join(path_of_features)
                print(path_of_features)
                np.save(path_of_features, bf.numpy())

            


    def load_id(self, split):
        path = (
            "./content/drive/My Drive/datasets/Flickr8k/Flickr8k_text/Flickr_8k."
            + split
            + "Images.txt"
        )
        with open(path, "r") as f:
            lines = f.read().splitlines()
        return lines

    def load_path(self, split_id):
        split_path = []
        for id_ in split_id:
            path = self.img_folder + id_
            split_path.append(path)
        return split_path

    def load_caption(self, split_id):
        table = str.maketrans("", "", string.punctuation)
        with open(
            "./content/drive/My Drive/datasets/Flickr8k/Flickr8k_text/Flickr8k.token.txt",
            "r",
        ) as f:
            lines = f.read().splitlines()
        map = dict()
        for line in lines:
            idx, content = line.split()[0], line.split()[1:]
            idx = idx[:-2]
            if idx not in split_id:
                continue
            content = [w.lower().translate(table) for w in content if w.isalpha()]
            content = "<start> " + " ".join(content) + " <end>"
            if idx not in map:
                map[idx] = []
            map[idx].append(content)
        return map

    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def tokenizing(self):
        all_train_captions = []
        for ele in self.train_captions.values():
            all_train_captions = all_train_captions + ele
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.top_k,
            oov_token="<unk>",
            filters= '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
        )

        self.tokenizer.fit_on_texts(all_train_captions)
        train_seqs = self.tokenizer.texts_to_sequences(all_train_captions)

        self.tokenizer.word_index["<pad>"] = 0
        self.tokenizer.index_word[0] = "<pad>"

        train_seqs = self.tokenizer.texts_to_sequences(all_train_captions)

        self.max_length = self.calc_max_length(train_seqs)

    def create_split(self, split_captions):
        img_to_cap_vector = collections.defaultdict(list)
        for img_id, img_cap in split_captions.items():
            img_to_cap_vector[img_id] = pad_sequences(
                self.tokenizer.texts_to_sequences(img_cap),
                padding="post",
                maxlen=self.max_length,
            )

        cap_split = []
        name_split = []
        for img_id in img_to_cap_vector.keys():
            cap_split.extend(img_to_cap_vector[img_id].tolist())
            name_split.extend([img_id] * 5)
        return cap_split, name_split

    def load_dataset(self, split):
        if split == "train":
            name_split, cap_split = self.name_train, self.cap_train
        else:
            name_split, cap_split = self.name_test, self.cap_test

        dataset = tf.data.Dataset.from_tensor_slices((name_split, cap_split))

        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(
                self.map_func, [item1, item2], [tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def map_func(self, img_id, cap):
        img_path = self.img_folder_np + img_id.decode("utf-8") + ".npy"
        img_tensor = np.load(img_path)
        return img_tensor, cap

