# AlgoTrading
## StockEnv - OpenAI gym environment

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
This agent is an example for how one can use the environment. It acts independently from the observations. The code for the agent:
```python
import random

class RandomAgent(Solver):
    def __init__(self, env):
        super().__init__(env)

    def _sample_action(self, obs):
        return random.choice([0,1,2])
```
### Moving Average Agent
The agent holds until the MA `window_size` and after enough samples it calculates the MA (`ma`) for that window. The agent buys/sells whenever the MA crosses the price. In other cases the agent holds.
