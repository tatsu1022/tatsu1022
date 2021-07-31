from .ReplayMemory import ReplayMemory
from .MultiStepBuffer import MultiStepBuffer


class MultiStepMemory(ReplayMemory):

    def __init__(self, state_shape, action_shape, userDefinedSettings):
        super(MultiStepMemory, self).__init__(state_shape, action_shape, userDefinedSettings)

        self.gamma = userDefinedSettings.gamma
        self.multi_step_reward_num = int(userDefinedSettings.multi_step_reward_num)
        if self.multi_step_reward_num != 1:
            self.buff = MultiStepBuffer(maxlen=self.multi_step_reward_num)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step_reward_num != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step_reward_num:
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                self.buff.reset()
        else:
            self._append(state, action, reward, next_state, done)
