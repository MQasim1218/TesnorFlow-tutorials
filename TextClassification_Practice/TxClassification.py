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
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import os
import string
import re

# ------------ HYPER-PARAMETERS --------------
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter, DatasetV1, DatasetV2

batch_size, seed = 32, 42  # Dataset Loading
val_set_div = 0.2
mx_tokens, seq_len = 10000, 250  # TextVectorization Layer
emb_dim = 16


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
print(os.listdir(test_dir))

# TODO: Reading a SampleFile
csharp_dir = os.path.join(train_dir, 'csharp')
print(csharp_dir)
s_file = os.path.join(csharp_dir, '1.txt')

print("Reading a Sample File")
with open(s_file) as f:
    print(f.read())


# TODO: Preparing data

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


raw_train_ds = prepare_ds(train_dir, batch_size, seed, val_set_div, 'training')
raw_val_ds = prepare_ds(train_dir, batch_size, seed, val_set_div, 'validation')
raw_test_ds = prepare_ds(test_dir, batch_size, seed)


# Use this function to check out a random sample
# Checkout_sample(raw_test_ds, 5)

# TODO: Data Preprocessing
# Standardize, Tokenize, Vectorize

# I will not be using this function as there are no HTML elements in the text that we pass
def cstm_stdfn(text):
    lc = tf.strings.lower(text)
    # the operation below strips out basic HTML
    formatted = tf.strings.regex_replace(lc, '<br />', '')
    return tf.strings.regex_replace(
        formatted, '[%s]' % re.escape(string.punctuation), ''
    )


def get_vecLayer(ds):
    print("Creating a text Vectorization layer using Keras.Layers.TextVectorizaton")
    VectLayer = tfk.layers.TextVectorization(
        standardize=cstm_stdfn,
        max_tokens=mx_tokens,
        output_mode='int',
        output_sequence_length=seq_len
    )
    text = ds.map(lambda x, y: x)
    print(f'Train text {text}')

    # Any of the datasets can be passed to adapt the layer to the shape of the incoming data
    # Making the Vectorization layer adapt to the inputs-Shape
    VectLayer.adapt(text)

    return VectLayer


VectorizationLayer = get_vecLayer(raw_train_ds)


def Vectorize(text, lbl):
    text = tf.expand_dims(input=text, axis=-1)
    return VectorizationLayer(text), lbl


# Testing the First vectorized data-batch
(C_txt, C_lbl) = next(iter(raw_train_ds))
f_txt, f_lbl = C_txt[0], C_lbl[0]

print("Printing out Information about the raw data and testing the Vectorization layer")
print(f"First text in the raw train dataset {f_txt}")
print(f"Label encoding of the raw train dataset {f_lbl}")
# print(f"Actual label {raw_train_ds.output_classes[f_lbl]}")
print(f"Printing the Vectorized text!!\n{Vectorize(f_txt, f_lbl)}")

AT = tf.data.AUTOTUNE
# Vectorizing the entire raw_train dataset
train_ds = raw_train_ds.map(Vectorize)
train_ds = train_ds.cache().prefetch(buffer_size=AT)

# Vectorizing the entire raw_validation dataset
val_ds = raw_val_ds.map(Vectorize)
val_ds = val_ds.cache().prefetch(buffer_size=AT)

# Vectorizing the entire raw_test dataset
test_ds = raw_test_ds.map(Vectorize)
test_ds = test_ds.cache().prefetch(buffer_size=AT)

#########################################################
################## Creating the Model ###################
#########################################################
'''
    Model Architecture
    Embeddings Layer -> Dropout -> GlobalAveragePooling -> Dropout -> Dense
'''
model = tfk.Sequential([
    tfk.layers.Embedding(mx_tokens + 1, emb_dim),
    tfk.layers.Dropout(0.2),
    tfk.layers.GlobalAveragePooling1D(),
    tfk.layers.Dropout(0.2),
    tfk.layers.Dense(4, activation=tfk.activations.relu)
], name="CoreClassifier")

model.summary()

model.compile(
    optimizer=tfk.optimizers.Adam(learning_rate=0.001),
    loss= tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics= ['accuracy']
)

mod_his = model.fit(train_ds, validation_data=val_ds, epochs=100, batch_size=64, verbose=1)
loss, acc = model.evaluate(test_ds, verbose=1, batch_size=64)

print(f"Model loss after training:: {loss}")
print(f"Model accuracy after training:: {acc}")

mod_dict = mod_his.history
print(f"{mod_dict.keys()}")

tr_acc = mod_dict['accuracy']
tr_loss = mod_dict['loss']

val_acc = mod_dict['val_accuracy']
val_loss = mod_dict['val_loss']


##################################################
###### Plotting Training / Validation Loss #######
##################################################
count = range(1,len(tr_acc) + 1)

plt.plot(count, tr_loss, 'bo', label='Training')
plt.plot(count, val_loss, 'b', label='Validation')
plt.title("Plotting Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


##################################################
###### Plotting Training / Validation Loss #######
##################################################
plt.plot(count, tr_acc, 'ro', label= 'Training')
plt.plot(count, val_acc, 'r', label= 'Validation')
plt.title("Plotting Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


################################################
# Creating a Model to Export
# CoreModel + Vectorization Layer + tf.nn.Sigmoid
exp_mod = tfk.Sequential([
    VectorizationLayer,
    model,
    tfk.layers.Activation('Softmax')
])

exp_mod.compile(
    loss= tfk.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tfk.optimizers.Adam(),
    metrics=['accuracy']
)

pr_loss, pr_acc = exp_mod.evaluate(raw_test_ds, verbose=1)
print(f'Prediction Accuracy of the Final Model:: {pr_acc}')
print(f'Prediction Loss of the Final Model:: {pr_loss}')

exp_data = [
    'The accuracy of the model is very low',
    'Loved all the scenes, will watch again with friends',
    'Horrible portrayal of the novel characters. The director hardly put any effort to match the novel, disgusted'
]

prediction = exp_mod.predict(exp_data)
for pr in prediction:
    print(f'Prediction{pr}')
    print(np.argmax(tf.nn.softmax(pr)))

print("End of Script")



