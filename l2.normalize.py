import sys
import numpy as np
import pandas as pd
import finplot as plt

import lib

data = pd.read_csv(
    sys.argv[1],
    header=None
)

# data = data.iloc[::10, :].reset_index(drop=True)
mid = (data[1] + data[3]) / 2

horizon = 200
n = 1000

data = lib.prepare(
    data,
    norm_window=n,
    predict_horizon=horizon,
    up_dn_threshold=0.5
)

labels = data['l']

mid = mid[n-1:-horizon].reset_index(drop=True)

times = data['t']

xs = times.astype('datetime64[ns]')

ax1, ax2, ax3 = plt.create_plot('', rows=3)

if True:
    plt.plot(xs, mid)
    plt.plot(xs, labels, color='#0a0', width=1.2, ax=ax3)
    plt.show()

data.to_csv(
    sys.stdout,
    float_format='%.7f',
    header=False,
    index=False
)
