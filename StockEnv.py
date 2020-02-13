import math
import gym
from gym import spaces, logger
import numpy as np
import pandas as pd
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
        1:      the agent sells with profit
    """
    def __init__(self, data_filter=None, max_length=365):
        # name of stocks to use
        self.data_filter = data_filter
        # max episode length
        self.max_length = max_length
        # stock data filenames
        f = []
        for (dirpath, dirnames, filenames) in walk(DATA_PATH):
            f.extend(filenames)
        self.fnames = f
        self.reset()

    def __read_data(self):
        if self.data_filter != None:
            fname = self.data_filter[np.random.randint(0, len(self.data_filter))]
        else:
            fname = self.fnames[np.random.randint(0, len(self.fnames))]
        csv = pd.read_csv(DATA_PATH + '/' + fname)
        seq = csv['Close'].to_numpy()
        self.seq = seq
        self.dates = csv['Date']

    def step(self, action):
        reward = 0
        self.last_action = action
        # buy
        if action == 1:
            n_available = int(self.m / self.seq[self.t])
            self.s += n_available
            self.m -= n_available * self.seq[self.t]
            self.bp = self.seq[self.t]
        # sell
        if action == 2:
            if self.s == 0:
                reward = -1
            else:                
                self.m += self.s * self.seq[self.t]
                self.s = 0
                reward = 1 if self.bp < self.seq[self.t] else -1 
        
        self.t += 1
        if self.t == self.max_length or self.t == self.seq.shape[0]-1:
            self.done = True
        self.last_reward = reward
        return np.array([self.seq[self.t]]), reward, self.done

    def reset(self):
        self.__read_data()

        # current timestep, current money, currently owned shares, buying price
        self.t, self.m, self.s, self.bp = 0, START_MONEY, 0, 0
        self.last_action, self.last_reward = 0, 0
        self.done = False

    def render(self):
        print('Date\t\tPrice\tNetWorth\tLastAction\tLastReward')
        print('%s\t%.3f\t%.2f \t%s\t\t%d' % (self.dates[self.t], self.seq[self.t], self.m + self.s*self.seq[self.t],
            ('hold', 'buy', 'sell')[self.last_action], self.last_reward))