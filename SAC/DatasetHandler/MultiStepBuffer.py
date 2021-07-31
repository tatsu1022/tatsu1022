from collections import deque
import numpy as np


class MultiStepBuffer:
    keys = ["state", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuffer, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
        }

    def append(self, state, action, reward):
        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([r * (gamma ** i) for i, r in enumerate(self.memory["reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuffer.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])
