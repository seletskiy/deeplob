import sys
import numpy as np
import pandas as pd
import finplot as plt

def prepare(
    data,
    norm_window=1000,
    predict_horizon=200,
    ewm_span=600,
    up_dn_threshold=0.5
):
    horizon = predict_horizon
    n = norm_window

    mid = (data[1] + data[3]) / 2

    ema = mid.ewm(span=ewm_span).mean()
    ema_horizon = ema.shift(-horizon)

    change_up = (ema_horizon - ema) >=  up_dn_threshold
    change_dn = (ema_horizon - ema) <= -up_dn_threshold

    def fill_gaps(x, state):
        if len(x) < 10:
            return False
        if state['n'] == 0:
            if x[0] == 0.0:
                return False
        else:
            if x[0] == 1.0:
                state['n'] += 1
                return True

        b = 0.0
        for i in range(1, 10):
            b = b + x[i]
        if b > 0 and state['n']+b >= 5:
            state['n'] += 1
            return True
        else:
            state['n'] = 0
            return False

    t = 10
    r = 20
    state = {'n': 0}
    change_up_new = change_up.rolling(r).apply(func=fill_gaps, raw=True, kwargs={'state': state})
    change_up_new = change_up_new.shift(-r+1)

    state = {'n': 0}
    change_dn_new = change_dn.rolling(r).apply(func=fill_gaps, raw=True, kwargs={'state': state})
    change_dn_new = change_dn_new.shift(-r+1)

    labels = ema.transform(lambda x: 1)

    labels.loc[change_up_new == True] = 2
    labels.loc[change_dn_new == True] = 0

    data = data[:-horizon]
    labels = labels[:-horizon]

    times = data.pop(0)

    avg = data.rolling(n).mean()
    std = data.rolling(n).std()

    avg = avg[n-1:]
    std = std[n-1:]

    c_p = data.columns[0::2]
    c_v = data.columns[1::2]

    avg_p = avg[c_p].mean(axis=1)
    avg_v = avg[c_v].mean(axis=1)

    # calculate total stdev over price and volume having separate stdevs over each price/volume columns
    std_p = (((n - 1) * std[c_p].pow(2).sum(axis=1) + n * (avg[c_p].sub(avg_p, axis=0)).pow(2).sum(axis=1)) / (n * len(c_p) - 1)).pow(0.5)
    std_v = (((n - 1) * std[c_v].pow(2).sum(axis=1) + n * (avg[c_v].sub(avg_v, axis=0)).pow(2).sum(axis=1)) / (n * len(c_v) - 1)).pow(0.5)

    data = data[n-1:]

    data[c_p] = data[c_p].sub(avg_p, axis=0).div(std_p, axis=0)
    data[c_v] = data[c_v].sub(avg_v, axis=0).div(std_v, axis=0)

    data.insert(0, 't', times)
    data.insert(1, 'l', labels)

    data = data.reset_index(drop=True)

    return data
