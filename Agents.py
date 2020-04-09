from Solver import Solver
from StockEnv import StockEnv
import random
import numpy as np

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