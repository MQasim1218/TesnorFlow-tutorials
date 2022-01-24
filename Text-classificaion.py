# Std Libraries...
import os
import re
import shutil
import string

# Data manipulation libraries...
import matplotlib.pyplot as pl
import numpy as np

# Deep-Learning libraries
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers as lrys, optimizers as opts, losses

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file(
                    "aclImdb_v1", url,
                    untar=True, cache_dir='.',
                    cache_subdir=''
                    )

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

# For data-preprocessing, it has to be passed onto library that expects a file structure like:
# main_directory/
# ...class_a/
# ......a_text_1.txt
# ......a_text_2.txt
# ...class_b/
# ......b_text_1.txt
# ......b_text_2.txt

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

### FunctionParameters ###
batch_size = 32
seed = 42

RawTrainDataset = tfk.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size= batch_size, seed= seed,
    validation_split= 0.2, subset= 'training'
)

# print(RawTrainDataset)
# Label 0 corresponds to neg
# Label 1 corresponds to pos


for txt_b, lbl_b in RawTrainDataset.take(1):
    for i in range(3, 5):
        print(f"Review:: {txt_b.numpy()[i]}")
        print(f"Review:: {lbl_b.numpy()[i]}")



print("All processes finished")
