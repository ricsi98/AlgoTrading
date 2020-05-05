from Solver import Solver
from StockEnv import StockEnv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomAgent(Solver):
    def __init__(self, env):
        super().__init__(env)

    def _sample_action(self, obs):
        return random.choice([0,1,2])

class MovingAverageAgent(Solver):
    def __init__(self, env, window_size):
        super().__init__(env)
        self.window_size = window_size

    def _sample_action(self, obs):
        if len(self.memory) < self.window_size + 1:
            return 0
        ma = self._ma(self.window_size)
        if obs[0] > ma and self.memory[-1][0][0] <= ma:
            return 1
        elif obs[0] < ma and self.memory[-1][0][0] >= ma:
            return 2
        return 0

    def _get_price_history(self, window_size):
        arr = []
        for i in range(window_size, 0, -1):
            arr.append(self.memory[-i][0][0])
        return arr

    def _ma(self, window_size):
        return np.ma.average(self._get_price_history(window_size))

class DqnAgent(nn.Module, Solver):
    def __init__(self, env, window_size, min_epsilon, ada_divisor):
        nn.Module.__init__(self)
        Solver.__init__(self, env)
        self.window_size = window_size
        self.memory = []
        self.gamma = 0.9
        self.hidden_size = 16

        self.t = 0
        self.min_epsilon = min_epsilon
        self.ada_divisor = ada_divisor
        self.hidden = None

        ## DQN
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1)
        self.linear = nn.Linear(self.hidden_size, 3)
        self.optimizer = torch.optim.SGD(self.parameters(), 0.001, 0.01)

    def remember(self, obs, action, reward, obs_, done):
        self.memory.append((obs, action, reward, obs_, done))

    def _init_h(self, batch_size=1):
        # (num_layers * num_directions, batch, hidden_size)
        return (torch.zeros((1, batch_size, self.hidden_size), dtype=torch.double, requires_grad=False), torch.zeros((1, batch_size, self.hidden_size), dtype=torch.double, requires_grad=False))

    def forward(self, x, hiddens):
        seq_len = x.shape[0]
        x, hiddens = self.lstm(x, hiddens)
        x = x.view(seq_len, 1, 1, self.hidden_size)[-1,-1,-1,:]
        x = F.relu(hiddens[0].view(-1))
        x = self.linear(x)
        return F.relu(x), hiddens
    
    def learn(self, optimizer):
        loss = nn.MSELoss()
        hiddens = self._init_h()
        for i, j in enumerate(self.memory):
            obs, action, reward, obs_, done = j
            obs = torch.tensor([[[obs[1]]]], dtype=torch.double)
            obs_ = torch.tensor([[[obs_[1]]]], dtype=torch.double)
            optimizer.zero_grad()
            y_, hiddens = self.forward(obs, hiddens)
            target = y_.clone()
            target[action] = reward if done else reward + self.gamma * torch.max(self.forward(obs_, hiddens)[0])
            target.detach()
            L = loss(y_, target)
            L.backward(retain_graph=True)
            optimizer.step()

    def get_action(self, obs, epsilon):
        if np.random.uniform() < epsilon:
            return 0 if np.random.uniform() >= 0.5 else 1
        if self.hidden == None:
            self.hidden = self._init_h()
        # (seq_len, batch, input_size)
        data = torch.tensor(obs.reshape(self.window_size, 1, 1), dtype=torch.double)
        x, self.hidden = self.forward(data, self.hidden)
        return torch.argmax(x)

    def epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - np.log10((t + 1) / self.ada_divisor)))

    def _get_price_history(self, window_size):
        arr = []
        for i in range(window_size, 0, -1):
            arr.append(self.memory[-i][0][1])
        return arr

    def _sample_action(self, obs):
        if len(self.memory) < self.window_size + 1:
            return 0
        self.t += 1
        hist = self._get_price_history(self.window_size-1)
        hist.append(obs[1])
        hist = torch.tensor(hist, dtype=torch.double)
        return self.get_action(hist, self.epsilon(self.t))

    def _after_episode(self):
        if self.render:
            print('Updating weights...')
        self.learn(self.optimizer)
        self.hidden = None
