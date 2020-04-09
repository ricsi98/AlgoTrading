import math
import gym
from gym import spaces, logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import walk

# path to folder where the csv-s are located
DATA_PATH = 'E:/stockdata/Stocks'
# starting money
START_MONEY = 10000

class StockEnv(gym.Env):
    """
    Description:
        The environment replays real stock data, where the agent can buy, hold and sell. The goal is to realize profit.

    Data source:
        https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data

    Observation:
        Type: Box(1)
        Num     Observation     Min     Max
        0       Close           0       inf

    Actions:
        Type: Discrete(3)
        Code        Action
        0           hold
        1           buy
        2           sell

    Reward:
        -1:     the agent tries to sell without buying / selling with negative profit
        profit: the agent sells with profit
    """
    def __init__(self, data_filter=None, max_length=100000, start_date=None, end_date=None):
        # name of stocks to use
        self.data_filter = data_filter
        # max episode length
        self.max_length = max_length
        # date interval
        self.start_date = start_date
        self.end_date = end_date

        self.buys, self.sells, self.holds = [], [], []
        self.rewards = 0
        # stock data filenames
        f = []
        for (dirpath, dirnames, filenames) in walk(DATA_PATH):
            f.extend(filenames)
        self.fnames = f
        self.reset()

    def _read_data(self):
        if self.data_filter != None:
            fname = self.data_filter[np.random.randint(0, len(self.data_filter))]
        else:
            fname = self.fnames[np.random.randint(0, len(self.fnames))]
        csv = pd.read_csv(DATA_PATH + '/' + fname)
        csv['Date'] = pd.to_datetime(csv['Date'])
        # filter data
        if self.start_date != None:
            csv = csv.loc[csv['Date'] > self.start_date]
        if self.end_date != None:
            csv = csv.loc[csv['Date'] < self.end_date]
        self.prices = csv['Close'].to_numpy().copy()
        seq = csv['Close'].to_numpy()
        self.seq = self._process(seq)
        self.dates = csv['Date'].to_numpy()

    def _process(self, data):
        for i in range(data.shape[0]-1, 1, -1):
            data[i] = data[i] / data[i-1]
        return data


    def step(self, action):
        reward = 0
        self.last_action = action
        # buy
        if action == 1:
            n_available = int(self.m / self.seq[self.t])
            self.s += n_available
            self.m -= n_available * self.seq[self.t]
            self.bp = self.seq[self.t]
            self.buys.append(self.t)
        # sell
        if action == 2:
            self.sells.append(self.t)
            if self.s == 0:
                reward = -1
            else:                
                self.m += self.s * self.seq[self.t]
                self.s = 0
                reward = self.s * self.seq[self.t]#1 if self.bp < self.seq[self.t] else -1 
        if action == 0:
            self.holds.append(self.t)
        self.t += 1
        if self.t == self.max_length or self.t == self.seq.shape[0]-1:
            self.done = True
        self.last_reward = reward
        self.rewards += reward
        return np.array([self.seq[self.t]]), reward, self.done

    def reset(self):
        self._read_data()

        # current timestep, current money, currently owned shares, buying price
        self.t, self.m, self.s, self.bp = 0, START_MONEY, 0, 0
        self.last_action, self.last_reward = 0, 0
        self.done = False

    def render(self):
        if self.t == 0: print('Date,Price,NetWorth,Balance,Action,LastReward')
        print('%s,%.1f,%.2f,%.2f,%s,%d' % (np.datetime_as_string(self.dates[self.t], unit='D'), 
            self.seq[self.t], self.m + self.s*self.seq[self.t],
            self.m, ('hold', 'buy', 'sell')[self.last_action], self.last_reward))

    def render_all(self):
        plt.plot(self.prices)
        plt.plot(self.buys, self.prices[self.buys], 'go')
        plt.plot(self.sells, self.prices[self.sells], 'ro')
        plt.plot(self.holds, self.prices[self.holds], 'yo')
        plt.title("roi: %.3f cum_rew: %.3f" % (float(self.m) / 10000.0, self.rewards))
        plt.show()