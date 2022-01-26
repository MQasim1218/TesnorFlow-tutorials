"""
    In this project, we are trying to classify coding examples as belonging to one of the four languages
    [ Python: 0, CSharp:1, Java:2, JavaScript:3 ]

    The Problem is a multi-classification problem and an extension to the last practice task done (binary
    classification of movie comments sentiment analysis on IMDB-ds )

    ------------- GOOD LUCK TO ME ------------
"""
# import tensorflow as tf
import tensorflow.keras as tfk
# import matplotlib.pyplot as plt
import os
# import re

# ------------ HYPER-PARAMETERS --------------
batch_size, seed = 32, 42
# ------------ HYPER-PARAMETERS --------------


def Download_dataset(url):
    ds = tfk.utils.get_file(
        "stack_overflow_16k", url,
        untar=True, cache_dir='.',
        cache_subdir=''
    )
    return ds


def prepare_ds(dir_name, b_s, s):
    return tfk.utils.text_dataset_from_directory(dir_name, batch_size=b_s, seed=s)


url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
dataset = Download_dataset(url)

# TODO: Loading the Dataset
# os.path.join() returns a concatenated path for a file, using an existing files path and the target files name...
dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k')
print(os.listdir(dataset_dir))

# TODO: Setting the Training files
train_dir = os.path.join(dataset_dir, 'train')
# Listdir lists all the files and subdirectories in the directory
print(os.listdir(train_dir))

# TODO: Reading a SampleFile
csharp_dir = os.path.join(train_dir, 'csharp')
print(csharp_dir)
s_file = os.path.join(csharp_dir, '1.txt')

print("Reading a Sample File")
with open(s_file) as f:
    print(f.read())

# Preparing data
raw_train_ds = prepare_ds(train_dir, batch_size, seed)
