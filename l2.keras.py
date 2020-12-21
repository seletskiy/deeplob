seed = 1338

import finplot as plt

import sys
import os
os.environ['PYTHONHASHSEED']=str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.utils import class_weight

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2), kernel_initializer='he_uniform')(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2), kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10), kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_uniform')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same', kernel_initializer='he_uniform')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)

    # use the MC dropout here
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm, kernel_initializer='he_uniform')(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax', kernel_initializer='he_uniform')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1)
    sgd = keras.optimizers.SGD(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

np.set_printoptions(precision=3, suppress=True)

T = 100
B = 64

data = pd.read_csv(
    sys.argv[1],
    header=None
)

prefix, _ = sys.argv[1].split('@', 2)

labels = data.pop(1)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels)

data = data[:-T]
labels = labels[T:]

times  = data.pop(0)

pivot = int(len(data) * 0.8)

data_train = data[:pivot]
data_probe = data[pivot:]

labels = tf.keras.utils.to_categorical(y=labels, num_classes=3)

labels_train = labels[:pivot]
labels_probe = labels[pivot:]

train = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data_train,
    targets=labels_train,
    sequence_length=T,
    sequence_stride=1,
    shuffle=False,
    batch_size=B)

probe = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data_probe,
    targets=labels_probe,
    sequence_length=T,
    sequence_stride=1,
    shuffle=False,
    batch_size=B)


if len(sys.argv) == 3:
    deeplob = load_model(sys.argv[2])
else:
    deeplob = create_deeplob(T, 40, 64)

deeplob.fit(
    train,
    validation_data=probe,
    epochs=400,
    class_weight=dict(enumerate(class_weights)),
    callbacks=[keras.callbacks.ModelCheckpoint(
        # "deeplob^globex:nqz0@{epoch}={val_accuracy:.3f}.h5",
        "deeplob^" + prefix + "@{epoch:03d}~{accuracy:.3f}..{val_accuracy:.3f}.h5",
        monitor='val_accuracy',
        mode='max',
        save_best_only=False
    )],
    verbose=1
)
