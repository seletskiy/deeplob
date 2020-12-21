import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import seaborn as sns
import pandas as pd
# import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

np.set_printoptions(precision=3, suppress=True)

################################################################################

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.show()

dataframe = pd.read_csv(
    "test.csv",
)

################################################################################

# seed = 1337

# dataframe_probe = dataframe.sample(frac=0.2, random_state=seed)
# dataframe_train = dataframe.drop(dataframe_probe.index)

# print("%d/%d" % (len(dataframe_train), len(dataframe_probe)))

# dataset_probe = dataframe_to_dataset(dataframe_probe)
# dataset_train = dataframe_to_dataset(dataframe_train)

# batch_size = 8

# batches_probe = dataframe_probe.batch(batch_size)
# batches_train = dataframe_train.batch(batch_size)

# inputs = []
# inputs_encoded = []

# for i in range(1, len(dataframe.columns)):
#     column = dataframe.columns[i]

#     feature = keras.Input(shape=(1,), name=column)

#     normalizer = Normalization()
#     normalizer.adapt(np.array(dataframe[column]))

#     inputs += [feature]
#     inputs_encoded += [normalizer(feature)]


epochs = 25
sequence_length = 10
batch_size = 32
pivot_split = 0.2

pivot = int(len(dataframe)*(1-pivot_split)) - 4

dataframe_features = dataframe.copy()
dataframe_labels = dataframe_features.pop('y')
dataframe_labels = tf.keras.utils.to_categorical(y=dataframe_labels, num_classes=3)

dataframe_sma_3 = dataframe_features.pop('sma_3').to_numpy()
# dataframe_sma_6 = dataframe_features.pop('sma_6').to_numpy()
dataframe_price = dataframe_features.pop('price').to_numpy()

# dataframe_sma_3 = dataframe_features.pop('sma_3').to_numpy()
# dataframe_sma_6 = dataframe_features.pop('sma_6').to_numpy()

dataframe_std = dataframe_features.std()
dataframe_mean = dataframe_features.mean()

dataframe_features = (dataframe_features - dataframe_std) / dataframe_mean

dataframe_features_train = dataframe_features[:pivot]
dataframe_features_probe = dataframe_features[pivot:]
dataframe_labels_train = dataframe_labels[:pivot]
dataframe_labels_probe = dataframe_labels[pivot:]

data_features_train = np.array(dataframe_features_train, dtype=np.float32)
data_labels_train = np.zeros(dataframe_labels_train.shape)
data_labels_train[:-sequence_length+1, :] = dataframe_labels_train[sequence_length-1:, :]

timeseries_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data_features_train,
    targets=data_labels_train,
    sequence_length=sequence_length,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size)

data_features_probe = np.array(dataframe_features_probe, dtype=np.float32)
data_labels_probe = np.zeros(dataframe_labels_probe.shape)
data_labels_probe[:-sequence_length+1, :] = dataframe_labels_probe[sequence_length-1:, :]

timeseries_probe = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data_features_probe,
    targets=data_labels_probe,
    sequence_length=sequence_length,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size)

timeseries = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=np.array(dataframe_features, dtype=np.float32),
    targets=None,
    sequence_length=sequence_length,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size)

model = tf.keras.Sequential([
    # normalizer,
    layers.Input(shape=(sequence_length, len(dataframe_features.columns))),
    layers.Flatten(),
    # layers.Dense(12, activation='relu'),
    # layers.Dense(12, activation='relu'),
    # layers.Dense(12, activation='relu'),
    # layers.Embedding(input_dim = 9, output_dim = 256, input_length = 10),
    # layers.LSTM(256, dropout = 0.3, recurrent_dropout = 0.3),
    # layers.GRU(256),
    # layers.Dense(2, activation="elu", kernel_regularizer=regularizers.l2(0.001)),
    # layers.Dense(2, activation="elu"),
    layers.Dense(512, activation="elu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    # layers.Dense(3, activation="softmax"),
    # layers.Dense(3, activation="elu"),
    # layers.Dense(3, activation="elu"),
    layers.Dense(3, activation="softmax"),
    # layers.Reshape([1, -1]),
])

# model.summary()

# def categorical_accuracy(actual, predicted):
#     predicted = tf.keras.backend.round(tf.keras.backend.clip(predicted, 0, 1))

#     actual = tf.cast(actual, tf.bool)
#     predicted = tf.cast(predicted, tf.bool)

#     return tf.math.logical_not(tf.math.logical_xor(actual, predicted))

# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#   0.001,
#   decay_steps=10*500,
#   decay_rate=1,
#   staircase=False)

# model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mean_squared_error", metrics=[categorical_accuracy])#, metrics=["accuracy"])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
# model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["accuracy"])
# model.compile(optimizer='SGD', loss="mean_squared_error")

# sns.pairplot(dataframe_train[['y']], diag_kind='kde')
show_heatmap(dataframe)

history = model.fit(
    timeseries_train,
    validation_data=timeseries_probe,
    # dataframe_features, dataframe_labels,
    # data_features, data_labels,
    epochs=epochs,
    # validation_split=validation_split,
    # batch_size=40,
    class_weight={
        0: 1,
        1: 8.95,
        2: 8.95
    },
    # validation_data=batches_probe,
    # callbacks=[keras.callbacks.ModelCheckpoint(
    #     "predict@{epoch}={val_accuracy:.3f}.h5",
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True
    # )],
)

model.summary()

# halt
# keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# f = np.vectorize(lambda x: 1 if x > 0.5 else 0)
# timeseries_probe_clean = timeseries_probe.map(lambda f, l: (f * dataframe_mean + dataframe_std, l))
# x = list(timeseries_probe_clean.as_numpy_iterator())

predicted = model.predict(timeseries)[:]

actual = dataframe_labels

fig, axs = plt.subplots(2, 2, sharex='col')

# print(tf.reduce_mean(
#     tf.cast(
#         tf.math.logical_not(tf.math.logical_xor(tf.cast(f(predicted), tf.bool), tf.cast(actual, tf.bool))),
#         tf.float64
#     )
# ))

# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.grid(True)

y_ticks = np.arange(0, len(actual))

axs[0, 0].grid(which='both')
axs[0, 0].axvline(x=pivot, color='r')

# dataframe_sma = dataframe_sma[pivot:]
# dataframe_sma_3 = dataframe_sma_3[pivot:]
# dataframe_sma_6 = dataframe_sma_6[pivot:]
# dataframe_price = dataframe_price[pivot:]

axs[0, 0].plot(y_ticks, dataframe_sma_3, color='#bb8cff', lw=1)
# axs[0, 0].plot(y_ticks, dataframe_sma_6, color='#03fc9d', lw=1)
# axs[0, 0].plot(y_ticks, dataframe_sma_3, color='b', lw=1)
# axs[0, 0].plot(y_ticks, dataframe_sma_6, color='r', lw=1)
axs[0, 0].plot(y_ticks, dataframe_price, color='#d6baff', lw=1)

# xx = dataframe['spy_sma_3_6_crossover'][pivot:].to_numpy()
# axs[0, 0].scatter(
#     [i for i in np.arange(0, len(actual)) if xx[i] == 1],
#     [dataframe_price[i] for i in np.arange(0, len(actual)) if xx[i] == 1],
# )

# xx = dataframe['spy_sma_3_6_crossunder'][pivot:].to_numpy()
# axs[0, 0].scatter(
#     [i for i in np.arange(0, len(actual)) if xx[i] == 1],
#     [dataframe_price[i] for i in np.arange(0, len(actual)) if xx[i] == 1],
# )

axs[0, 0].scatter(
    [i for i in np.arange(0, len(actual)) if np.argmax(actual[i]) == 1],
    [dataframe_price[i] for i in np.arange(0, len(actual)) if np.argmax(actual[i]) == 1],
    color='g',
    marker='^',
    s=20
)

axs[0, 0].scatter(
    [i for i in np.arange(0, len(actual)) if np.argmax(actual[i]) == 2],
    [dataframe_price[i] for i in np.arange(0, len(actual)) if np.argmax(actual[i]) == 2],
    color='g',
    marker='v',
    s=20
)

axs[0, 0].scatter(
    [i for i in np.arange(0, len(predicted)) if np.argmax(predicted[i]) == 1],
    [dataframe_price[i] - 1 for i in np.arange(0, len(predicted)) if np.argmax(predicted[i]) == 1],
    color='r',
    marker='^',
    s=20
)

axs[0, 0].scatter(
    [i for i in np.arange(0, len(predicted)) if np.argmax(predicted[i]) == 2],
    [dataframe_price[i] + 1 for i in np.arange(0, len(predicted)) if np.argmax(predicted[i]) == 2],
    color='r',
    marker='v',
    s=20
)
axs[0, 0].grid(True)

# axs[0, 0].scatter(
#     [i for i in np.arange(0, len(predicted)) if predicted[i] > 0.33],
#     [dataframe_price[i] - 1 for i in np.arange(0, len(predicted)) if predicted[i] > 0.33],
#     color='r',
#     marker='^',
#     s=20
# )

# axs[0, 0].scatter(
#     [i for i in np.arange(0, len(predicted)) if predicted[i] < -0.25],
#     [dataframe_price[i] + 1 for i in np.arange(0, len(predicted)) if predicted[i] < -0.25],
#     color='r',
#     marker='v',
#     s=20
# )

axs[1, 0].plot(np.arange(len(predicted)), [[0, 1, -1][i] for i in np.argmax(predicted, axis=1)], color='gray', lw=1)
axs[1, 0].scatter(np.arange(len(actual)), [[0, 1, -1][i] for i in np.argmax(actual, axis=1)], s=1, color='black')
axs[1, 0].grid(True)

# axs[1, 0].set_ylim(-3, 3)
# axs[1, 0].set_xlim(1200, 1500)

axs[0, 1].plot(history.history['loss'], label='loss')
axs[0, 1].plot(history.history['val_loss'], label='val_loss')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 1].plot(history.history['accuracy'], label='accuracy')
axs[1, 1].plot(history.history['val_accuracy'], label='val_accuracy')
axs[1, 1].legend()
axs[1, 1].grid(True)

# print(model.layers[1].get_weights())
# axs[1, 1].bar(x = range(len(dataframe_features.columns)), height=model.layers[1].kernel[:,0].numpy())
# axis = axs[1, 1].gca()
# axis.set_xticks(range(len(dataframe_features.columns)))
# _ = axis.set_xticklabels(dataframe_features.columns, rotation=90)

plt.tight_layout(pad=0)
plt.show()

# print("dataset[0]", dataset[0])

# sample = {
#     "x": 60,
# }

# input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
# predictions = model.predict(input_dict)

# print("predictions", predictions)

# print(
#     "This particular patient had a %.1f percent probability "
#     "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],)
# )
