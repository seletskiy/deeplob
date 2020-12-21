import sys
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import finplot as plt
import lib

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

seed = 1338

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
np.set_printoptions(precision=3, suppress=True)

T = 100
B = 64

horizon = 200
n = 1000

data = pd.read_csv(
    sys.argv[1],
    header=None
)

# data = data.fillna(0).tail(-100)
# data = data.iloc[:10000, :]
# data = data.reset_index(drop=True)

mid = (data[1] + data[3]) / 2

data = lib.prepare(
    data,
    norm_window=n,
    predict_horizon=horizon,
    up_dn_threshold=0.5
)

mid = mid[n-1:-T-horizon].reset_index(drop=True)
data = data[:-T]

times  = data.pop('t').astype('datetime64[ns]')
labels = data.pop('l')

test = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data,
    # targets=tf.keras.utils.to_categorical(y=labels, num_classes=3),
    targets=None,
    sequence_length=T,
    sequence_stride=1,
    shuffle=False,
    batch_size=B)

deeplob = tf.keras.models.load_model(sys.argv[2])

predictions = deeplob.predict(test)
predictions = pd.Series(np.argmax(predictions, axis=1))

mid = mid[T-1:].reset_index(drop=True)
xs = times[T-1:].reset_index(drop=True)
labels = labels[T-1:].reset_index(drop=True)

ax1, ax2, ax3 = plt.create_plot('l2.plot.py', rows=3)

plt.plot(xs, mid, legend='price', ax=ax1)
plt.plot(xs, predictions, legend='prediction', ax=ax2)
plt.plot(xs, labels, legend='actual', ax=ax3)

plt.show()
