# AlgoTrading
## StockEnv - OpenAI gym environment

### Description
The environment replays real stock data, where the agent can buy, hold and sell. The goal is to realize profit.

### Data source
[Price volume data for all us stocks etfs](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data)

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
