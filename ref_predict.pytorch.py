import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import torch.optim.lr_scheduler
# from torch.optim.lr_scheduler import ReduceLROnPlateau

seed = 1111

# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
# import sklearn as sk
import sklearn.preprocessing as skp

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class LSTM(torch.nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.seq_length = seq_length
        self.dropout = torch.nn.Dropout(p=0.2)

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
                            # num_layers=num_layers, batch_first=True,dropout = 0.25)

        self.fc = torch.nn.Linear(hidden_size, num_classes)

        # torch.nn.init.xavier_normal_(self.lstm.all_weights)

        # for name, param in self.named_parameters():
        #     torch.nn.init.xavier_normal_(param.data)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)

        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))

        h_out = h_out.view(self.num_layers, x.size(0), self.hidden_size)[-1]

        out = self.fc(h_out)
        # out = self.dropout(out)

        return out


data_source = pd.read_csv(
    "test.csv",
    usecols=[
        'y',
        # 'stddev_3',
        'ref_0_open',
        'ref_0_low',
        'ref_0_high',
        'ref_0_vol',
        'ref_1_open',
        'ref_1_close',
        'ref_1_vol',
        'ref_2_open',
        'ref_2_close',
        'ref_2_vol',
        'ref_3_open',
        'ref_3_close',
    ],
)

def create_batches(dataset, window_size, step=0):
    x, y = list(), list()
    for i in range(0, len(dataset), int(window_size / 2) if step == 0 else step):
        end = i + window_size
        if end > len(dataset):
            break
        seq_x, seq_y = dataset[i:end, 1:], dataset[end-1, 0]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

# data_source = pd.read_csv(
#     "test.sum.csv",
#     usecols=['sum', 'a', 'b'],
# )

columns = list()
scalers = list()

for col_name in data_source.columns:
    print("col_name", col_name)
    scaler = skp.MinMaxScaler(feature_range=(-1, 1))

    columns += [
        scaler.fit_transform(
            data_source[col_name].values.astype(float).reshape(-1, 1)
        )
    ]

    scalers += [scaler]

data = np.hstack(columns)

features_count = len(columns) - 1
batch_size = 60

pivot = int(len(data) * 0.67)
data_teach = data[:pivot]
data_probe = data[pivot:]

# model = LSTM(features_count, batch_size).cuda()
model = LSTM(1, features_count, 512, 3).cuda()
# model.apply(init_weights)

x, y = create_batches(data_teach, batch_size)

criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# scheduler = StepLR(optimizer, step_size=100, gamma=0.96)
scheduler_annealing = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, threshold=1e-4, patience=20)

scheduler = scheduler_annealing

epochs = 20000
batches_count = len(x)
# batches_count = int(len(x)/4)

# if False:
if True:
    model.train()

    for t in range(epochs):
        for b in range(0, len(x), batches_count):
            x_batch = x[b:b+batches_count, :, :]
            y_batch = y[b:b+batches_count]

            # print("x_batch", x_batch)
            # print("y_batch", y_batch)

            # print("x_batch.shape", x_batch.shape)

            x_batch_tensor = torch.tensor(x_batch, dtype=torch.float32)
            y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32)

            # model.init_hidden(x_batch_tensor.size(0))

            output = model(x_batch_tensor)

            # xxx_actual = scalers[0].inverse_transform(y_batch.reshape(-1, 1))
            # xxx_predicted = scalers[0].inverse_transform(output.view(-1).detach().cpu().reshape(-1, 1))

            # print("xxx_predicted", xxx_predicted)

            # print(np.hstack((xxx_actual, xxx_predicted)))

            loss = criterion(output.view(-1), y_batch_tensor)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if loss.item() < 0.00000100:
            scheduler = scheduler_plateau

        print('%5d %.10f' % (t, loss.item()))

        scheduler.step(loss)

    torch.save(model.state_dict(), 'test.pt')

model.load_state_dict(torch.load('test.pt'))

model.eval()

# print("len(data_probe)", len(data_probe))
# data_probe = data_probe
# data_probe = data_teach
# data_probe = data[690:718]
data_probe = data

x, y = create_batches(data_probe, batch_size, 1)

x_batch = torch.tensor(x, dtype=torch.float32)
# model.init_hidden(x_batch.size(0))

output = model(x_batch).cpu().data.numpy()

# input_low = scalers[1].inverse_transform(np.array(x[:, :, 0]))
# input_high = scalers[2].inverse_transform(np.array(x[:, :, 1]))

input_low = scalers[1].inverse_transform(np.array(x[:, -1, 0]).reshape(-1, 1))

# print("input_low", input_low)

predicted = scalers[0].inverse_transform(np.array(output).reshape(-1, 1))
# print("predicted", predicted)
actual = scalers[0].inverse_transform(np.array(y).reshape(-1, 1))
# print("actual", actual)

# volatility = scalers[1].inverse_transform(np.array(data_probe[:, 1]).reshape(-1, 1))

# print("input_low[-1]", input_low[-1])
# print("input_high[-1]", input_high[-1])
# print("actual[-1]", actual[-4:][:, 0])
# print("predicted[-4:][:, 0]", predicted[-4:][:, 0])

# mtp.grid(True, markevery=5, which='minor')

fig, ax_1 = mtp.subplots()
fig.tight_layout()

ax_1.grid(which='both')
# ax_1.set_xticks(np.arange(0, len(predicted), 5))

# mtp.autoscale(axis='x', tight=True)
ax_1.plot(np.arange(0, len(predicted)), predicted, 'r|-', ms=4, lw=1)
ax_1.plot(np.arange(0, len(actual)), actual, 'g|-', ms=4, lw=1)
# ax_1.tick_params(axis='y', color='tab:red')

# ax_2 = ax_1.twinx()

# ax = mtp.figure().add_subplot(1, 1, 1)
# ax_2.plot(np.arange(0, len(volatility)), volatility, 'b', markersize=2)
# ax.set_xticks(np.arange(0, len(predicted), 5), minor=True)
# ax.grid(which='both', alpha=0.2, linestyle=':')

mtp.show()
