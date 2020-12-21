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
    sys.argv[2],
    header=None
)

mid = (data[1] + data[3]) / 2

data = lib.prepare(
    data,
    norm_window=n,
    predict_horizon=horizon,
    up_dn_threshold=0.5
)

labels = data.pop('l')

times = data.pop('t').astype('datetime64[ns]')
times = times[T-1:].reset_index(drop=True)

data = data[:-T]
labels = labels[T-1:].reset_index(drop=True)

mid = mid[n-1:-T-horizon].reset_index(drop=True)
mid.name = 'mid'

test = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data,
    # targets=tf.keras.utils.to_categorical(y=labels, num_classes=3),
    targets=None,
    sequence_length=T,
    sequence_stride=1,
    shuffle=False,
    batch_size=B)

deeplob = tf.keras.models.load_model(sys.argv[1])

predictions = deeplob.predict(test)

is_bear_ticks = 0
is_bull_ticks = 0
is_neut_ticks = 0

position = 0
price = 0
profit = 0
trades_total = 0
trades_wins = 0
change = 0

trades = pd.DataFrame(index=times.index).reset_index(drop=True)

# mid = mid[norm_window:-T].reset_index(drop=True)
# times = pd.to_datetime(times[norm_window:-T].reset_index(drop=True), infer_datetime_format=True)
# mid.name = 'mid'
# times.name = 'time'

threshold = 20

def fee(qty, price):
    ibk = min(max(0.35, 0.0035*abs(qty)), 0.01*(abs(qty)*price))
    trx = 0
    act = 0
    if qty <= 0:
        trx = 0.0000221 * abs(qty) * price
        act = 0.000119 * abs(qty)
    nyse = 0.000175 * ibk
    finra = 0.00056 * ibk

    total = ibk + trx + act + nyse + finra

    print("total", total)

    return 4

lever = 20

for i in range(len(predictions)):
    prediction = np.argmax(predictions[i])

    is_bear = prediction == 0
    is_bull = prediction == 2

    if is_bear:
        is_bull_ticks = 0
        is_bear_ticks += 1
        is_neut_ticks = 0
    elif is_bull:
        is_bull_ticks += 1
        is_bear_ticks = 0
        is_neut_ticks = 0
    else:
        is_bear_ticks = 0
        is_bull_ticks = 0
        is_neut_ticks += 1

    close = False

    if is_bear_ticks >= threshold:
        if position == 0:
            position = -1 * lever
            price = mid[i]
            print((times[i], labels[i], position, price))
            trades.loc[i, 'shorts'] = price + 0.02

        if position > 0:
            close = True

    if is_bull_ticks >= threshold:
        if position == 0:
            position = 1 * lever
            price = mid[i]
            print((times[i], labels[i],  position, price))
            trades.loc[i, 'longs'] = price - 0.02

        if position < 0:
            close = True

    if close:
        change = (mid[i] - price) * position - fee(position, price)
        print((times[i], change))
        position = 0
        trades_total += 1
        if change > 0:
            trades_wins += 1
        profit += change

print("profit", profit)
print("trades_total", trades_total)
print("trades_wins/trades_total", trades_wins/trades_total)

xs = times
print("len(xs)", len(xs))
print("len(labels)", len(labels))
mid = mid[T-1:].reset_index(drop=True)
# labels = labels[T-1:].reset_index(drop=True)

ax1, ax2, ax3 = plt.create_plot('', rows=3)

plt.plot(xs, mid, legend='price', ax=ax1)

shorts = trades['shorts']
plt.plot(xs, shorts, ax=ax1, style='v', legend='short', color='#f00', width=2)

longs = trades['longs']
plt.plot(xs, longs, ax=ax1, style='^', legend='long', color='#0f0', width=2)

plt.plot(xs, labels, legend='actual', ax=ax2)

plt.plot(xs, pd.Series(np.argmax(predictions, axis=1)), legend='prediction', ax=ax3)

plt.show()
