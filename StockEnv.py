import math
import gym
from gym import spaces, logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import walk

# path to folder where the csv-s are located
DATA_PATH = '/media/ricsi98/HDD2/stockdata/'
# number of shares bought per buy action
BUY_COUNT = 1

class StockEnv(gym.Env):
    """
    Description:
        The environment replays real stock data, where the agent can buy, hold and sell. The goal is to realize profit.

    Data source:
        https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data

    Observation:
        Type: Box(2)
        Num     Observation     Min     Max
        0       Close           0       inf
        1       p[n+1] / p[n]   0       inf

    Actions:
        Type: Discrete(3)
        Code        Action
        0           hold
        1           buy
        2           sell

    Reward:
        -1:     the agent tries to sell without owning any stocks
        x:      x = (price[sell] - price[last_buy]) / price[last_buy] after n buys and a sell (n >= 1)        
    """
    def __init__(self, data_filter=None, max_length=100000, start_date=None, end_date=None):
        # name of stocks to use
        self.data_filter = data_filter
        # max episode length
        self.max_length = max_length
        # date interval
        self.start_date = start_date
        self.end_date = end_date

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
            csv = csv.loc[csv['Date'] > np.datetime64(self.start_date)]
        if self.end_date != None:
            csv = csv.loc[csv['Date'] < np.datetime64(self.end_date)]
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
        
        if action == 0:
            if self.s > 0:
                pass#reward = (self.seq[self.t] - 1)
            self.holds.append(self.t)
        elif action == 1:
            self.s += 1
            self.bp = self.prices[self.t]
            self.out_money += self.prices[self.t]
            self.buys.append(self.t)
        elif action == 2:
            if self.s < 1:
                reward = -1
            else:
                reward = (self.prices[self.t] / self.bp - 1) * self.s * 10
                self.in_money += self.prices[self.t] * self.s
                self.s = 0
            self.sells.append(self.t)
        
        self.last_reward = reward
        self.rewards += reward
        self.t += 1
        return (self.prices[self.t], self.seq[self.t]), reward, self.t >= len(self.prices)-1
            

    def reset(self):
        self._read_data()

        # current timestep, currently owned shares, buying price
        self.t, self.s, self.bp = 0, 0, 0
        self.buys, self.sells, self.holds = [], [], []
        self.last_action, self.last_reward = 0, 0
        self.out_money, self.in_money = 0, 0
        self.done = False
        self.rewards = 0
        return (self.prices[0], self.seq[0]), 0, False

    def render(self):
        if self.t == 0: print('Date,Price,NetWorth,Balance,Action,LastReward')
        print('%s,%.1f,%s,%.3f' % (np.datetime_as_string(self.dates[self.t], unit='D'), 
            self.seq[self.t], ('hold', 'buy', 'sell')[self.last_action], self.last_reward))

    def _calc_roi(self):
        return (self.in_money - self.out_money) / self.out_money

    def render_all(self):
        plt.plot(self.prices)
        plt.plot(self.buys, self.prices[self.buys], 'go', alpha=0.6)
        plt.plot(self.sells, self.prices[self.sells], 'ro', alpha=0.6)
        #plt.plot(self.holds, self.prices[self.holds], 'bo')
        plt.title("cumulative reward: %.3f roi %.3f" % (self.rewards, 1 + self._calc_roi()))
        plt.show()