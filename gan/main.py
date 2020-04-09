import models
import torch
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'E:/stockdata/Stocks'


def read_data(fname, start_date=None, end_date=None):
    csv = pd.read_csv(DATA_PATH + '/' + fname)
    csv['Date'] = pd.to_datetime(csv['Date'])
    # filter data
    if start_date != None:
        csv = csv.loc[csv['Date'] > start_date]
    if end_date != None:
        csv = csv.loc[csv['Date'] < end_date]
    return torch.tensor(csv['Close'].to_numpy(), dtype=torch.float64)

def prepare_batch(n_batches, max_len=50, offset=0):
    data = read_data('AAPL.us.txt')

    max_batch_size = int(data.shape[0] / max_len)
    if n_batches > max_batch_size or n_batches == -1:
        batch_size = max_batch_size

    placeholder = torch.zeros((max_len, batch_size, 1), dtype=torch.float64)
    for bs in range(batch_size):
        placeholder[:,bs,:] = data[bs * max_len + offset : (bs + 1) * max_len + offset].view(max_len, 1)
    return placeholder

def pretrain_gen(generator, opt):
    D1, D2 = prepare_batch(-1), prepare_batch(-1, offset=1)
    rewards = torch.ones((D1.shape[0],1))

    opt.zero_grad()
    pg_loss = generator.PGLoss(D1, D2, rewards)
    pg_loss.backward()
    opt.step()
    return pg_loss.item()


if __name__ == "__main__":
    gn = models.G_network(10)
    gn.double()

    opt = torch.optim.Adam(gn.parameters(), 0.001)

    for i in range(1000):
        print(pretrain_gen(gn, opt))

    

    x = gn.sample(1,300,250).numpy()[0,:]
    plt.plot(x)
    plt.show()