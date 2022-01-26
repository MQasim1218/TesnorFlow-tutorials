"""
    In this project, we are trying to classify coding examples as belonging to one of the four languages
    [ Python: 0, CSharp:1, Java:2, JavaScript:3 ]

    The Problem is a multi-classification problem and an extension to the last practice task done (binary
    classification of movie comments sentiment analysis on IMDB-ds )

    ---------------- Workflow ----------------
    -- Import Libraries
    1. Download Data
    2. Load Data (train, validation and test sets)
    3. Preprocess data (Create a pipeline for)
    {
        Standardize text (remove punctuation, code snippets etc.)
        Tokenize (break large text to smaller pieces)
        Vectorize (Convert text to int tensors)
    }
    4. Create text-embeddings layer to preprocess data
    5. Model building and training
    6. Model evaluation and hyperparameter tuning
    7. Wrap model into an outer model, along with the embeddings layer
    8. Make predictions and test results
    9. Chill
    ---------------- Workflow ----------------



    ------------- GOOD LUCK TO ME ------------
"""
import tensorflow as tf
import tensorflow.keras as tfk
# import matplotlib.pyplot as plt
import os
import string
import re

# import re

# ------------ HYPER-PARAMETERS --------------
batch_size, seed = 32, 42 # Dataset Loading
mx_tokens, seq_len = 10000, 250 # TextVectorization Layer

# ------------ HYPER-PARAMETERS --------------


def Download_dataset(data_url):
    ds = tfk.utils.get_file(
        "stack_overflow_16k", data_url,
        untar=True, cache_dir='.',
        cache_subdir=''
    )
    return ds


def Checkout_sample(ds, n):
    for txt, lbl in ds.take(1):
        print(f'Sample Text:: {txt.numpy()[n]}')
        print(f'Sample Review:: {lbl.numpy()[n]}')


def prepare_ds(dir_name, b_s, s, val_sp=0.0, subset=''):
    if val_sp != 0:
        ds = tfk.utils.text_dataset_from_directory(
            dir_name,
            batch_size=b_s,
            seed=s,
            subset=subset,
            validation_split=val_sp
        )
    else:
        ds = tfk.utils.text_dataset_from_directory(
            dir_name,
            batch_size=b_s,
            seed=s
        )
    return ds

# I will not be using this function as there are no HTML elements in the text that we pass
def cstm_stdfn(text):
    lc = tf.strings.lower(text)
    # the operation below strips out basic HTML
    formatted = tf.strings.regex_replace(lc, '<br />', '')
    return tf.strings.regex_replace(
        formatted, '[%s]' % re.escape(string.punctuation), ''
    )

url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
dataset = Download_dataset(url)

# TODO: Loading the Dataset
# os.path.join() returns a concatenated path for a file, using an existing files path and the target files name...
dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k')
print(os.listdir(dataset_dir))

# TODO: Setting the Training files
# Loading the path of test and training files...
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
print('Lists all the files and subdirectories in the Train directory')
print(os.listdir(train_dir))
print('Listdir lists all the files and subdirectories in the Test directory')

# TODO: Reading a SampleFile
csharp_dir = os.path.join(train_dir, 'csharp')
print(csharp_dir)
s_file = os.path.join(csharp_dir, '1.txt')

print("Reading a Sample File")
with open(s_file) as f:
    print(f.read())

# TODO: Preparing data
val_set_div = 0.2
raw_train_ds = prepare_ds(train_dir, batch_size, seed, val_set_div, 'training')
raw_val_ds = prepare_ds(train_dir, batch_size, seed, val_set_div, 'validation')
raw_test_ds = prepare_ds(test_dir, batch_size, seed)


# Use this function to check out a random sample
# Checkout_sample(raw_test_ds, 5)

# TODO: Data Preprocessing
# Standardize, Tokenize, Vectorize

print("Creating a text Vectorization layer using Keras.Layers.TextVectorizaton")
VectorizationLayer = tfk.layers.TextVectorization(
    max_tokens= mx_tokens,
    output_mode= 'int',
    output_sequence_length = seq_len
)
