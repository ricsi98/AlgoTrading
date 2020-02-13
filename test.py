from StockEnv import StockEnv
import numpy as np

class RandomAgent():
    def __init__(self, n_episodes=10):
        self.n_episodes = n_episodes
        self.env = StockEnv(data_filter=['aapl.us.txt', 'msft.us.txt'], max_length=100000)

    def run(self):
        for e in range(self.n_episodes):
            current_state = self.env.reset()
            done = False

            while not done:
                self.env.render()
                action = np.random.randint(0,3)
                obs, reward, done = self.env.step(action)

if __name__ == "__main__":
    solver = RandomAgent(1)
    solver.run()