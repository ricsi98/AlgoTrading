# AlgoTrading
## StockEnv - OpenAI gym environment

### Description
The environment replays real stock data, where the agent can buy, hold and sell. The goal is to realize profit.

### Data source
[Price volume data for all us stocks etfs](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data)
. Download data and update `DATA_PATH` variable ([StockData.py](../StockEnv)) to the path to the csv files.

### Observation
Type: Box(1)

Num | Observation | Min | Max
---|---|---|---
0 | Close Price of the stock | 0 | inf

### Actions
Type: Discrete(3)

Code | Action | Possible Rewards
---|---|---
0 | hold | 0
1 | buy  | 0
2 | sell | 1,-1

### Reward
+ -1: the agent tries to sell without buying / selling with negative profit
+ 1: the agent sells with profit

## Models
### Random Agent
This agent is an example for how one can use the environment. It acts independently from the observations. The code for the agent:
```python
from StockEnv import StockEnv
import numpy as np

# Create environment
env = StockEnv(data_filter=['aapl.us.txt', 'msft.us.txt'], max_length=100000)
env.reset()

done = False
while not done:
    env.render()
    # sample random action
    action = np.random.randint(0,3)
    obs, reward, done = env.step(action)
```
### Moving Average Agent
The agent holds until the MA `window_size` and after enough samples it calculates the MA (`ma`) for that window. The model buys/sells whenever the MA crosses the price:
```python
if price[t-1] < ma and price[t] > ma:
    # buy
elif price[t-1] > ma and price[t] < ma:
    # sell
else:
    hold
```
(Note: in my implementation I used a smaller window size for selling)
