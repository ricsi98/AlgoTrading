class Solver:
    def __init__(self, env, render=True):
        self.env = env
        self.render = render

    def _before_episode(self):
        pass

    def _after_episode(self):
        pass

    def _sample_action(self, obs):
        raise('implement solver._sample_action!')
        return None

    def run(self, episodes):
        for e in range(episodes):
            done = False
            obs, reward, done = self.env.reset()
            self.memory = []
            self._before_episode()
            while not done:
                if self.render:
                    self.env.render()
                action = self._sample_action(obs)
                obs_, reward, done = self.env.step(action)
                self.memory.append((obs, action, reward, obs_, done))
                obs = obs_
            self._after_episode()
            if self.render:
                self.env.render_all()