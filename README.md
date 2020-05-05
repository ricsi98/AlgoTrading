## Environment

### Description
The environment replays real stock data, where the agent can buy, hold and sell. The goal is to realize profit.

### Data source
[Price volume data for all us stocks etfs](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data)
. Download data and update `DATA_PATH` variable ([StockEnv.py](../StockEnv.py)) to the path of the csv files.

### Observation
Type: Box(2)

Num | Observation | Min | Max
---|---|---|---
0 | Close Price of the stock | 0 | inf
1 | p[n+1] / p[n] | 0 | inf

### Actions
Type: Discrete(3)

Code | Action | Possible Rewards
---|---|---
0 | hold | 0
1 | buy  | 0
2 | sell | 1,-1

### Reward
+ -1: the agent tries to sell without owning any stocks
+ x: (p[sell] - p[last_buy]) / p[last_buy] after n buys and a sell (n >= 1)

## Solver
You can implement your own trading strategy by inheriting from the `Solver` class. In this class you can interact with the environment by the following functions:
+ `_sample_action` this function takes the latest observation as parameter and has to return an action {0, 1, 2}
+ `_before_episode` this function runs before each episode
+ `_after_episode` this function runs after each episode

At any time, you can access all the previous observations and actions from the `memory` attribute. This list contains a (observation, action, reward, next_observation, done) tuple for every previous timesteps. (Note: this list is always one step behind, except in the _after_episode function. The most recent observation is passed by the `obs` parameter)

## Agents
### Random Agent
This agent is an example of how one can use the environment. It acts independently from the observations. The code for the agent:
```python
import random

class RandomAgent(Solver):
    def __init__(self, env):
        super().__init__(env)

    def _sample_action(self, obs):
        return random.choice([0,1,2])
```
### Moving Average Agent
The agent holds until it has enough items in memory (`window_size` iterations). After enough samples it calculates the MA. The agent buys/sells whenever the MA crosses the price from below/above. In other cases the agent holds.

Agent performance on appl.us with `window_siz` = 50:

<p align="center">
    <img src="https://github.com/ricsi98/AlgoTrading/blob/master/images/aapl_MA_50.png"/>    
</p>


### Deep Q Network Agent
This agent tries to learn patterns in the stock price movement. After many episodes it will be able to gain more and more profit (theoretically).
During training the model either acts greedyly or randomly.
Let `epsilon` be the probability of the model acting randomly. Then `epsilon` is given by *max(epsilon_min, min(1, 1 - log((t + 1) / ADA_DIVISOR)))* at any timestep *t*. This is a decreasing function with *epsilon_min* minimum. The *ADA_DIVISOR* determines the "speed" of the decrease, higher *ADA_DIVISOR* results higher probability of random acts at timestep *t*.
After each episode the model *"rewinds"* the whole episode, and for each timestep it minimizes the difference between its prediction *E(reward | state, action)* and the actual reward.
An example script for training this agent:
```python
from StockEnv import StockEnv
from Agents import DqnAgent
import torch

env = StockEnv(data_filter=['aapl.us.txt'])
agent = DqnAgent(env=env, window_size=30, min_epsilon=0.05, ada_divisor=50).double()
opt = torch.optim.Adam(agent.parameters(), lr=0.001)
agent.optimizer = opt
agent.run(100)
```
