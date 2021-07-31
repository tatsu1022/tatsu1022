import gym
import time

from .InterfaceEnvironment import InterfaceEnvironment


class RobotPush(InterfaceEnvironment):
    def __init__(self, userDefinedSettings):
        self.env = gym.make('FetchPush-v1')
        self.RENDER_INTERVAL = userDefinedSettings.RENDER_INTERVAL

    def get_state_action_space(self):
        STATE_DIM = self.env.observation_space['observation'].shape
        ACTION_DIM = self.env.action_space.shape
        return STATE_DIM, ACTION_DIM

    def reset(self):
        state = self.env.reset()['observation']
        return state

    def step(self, action):
        ret = self.env.step(action)
        next_state  = ret[0]['observation']
        reward, done, info = ret[1:]
        return next_state, reward, done, info

    def get_max_episode_steps(self):
        return self.env._max_episode_steps

    def random_action_sample(self):
        action = self.env.action_space.sample()
        return action

    def render(self):
        self.env.render()
        time.sleep(self.RENDER_INTERVAL)

    def close(self):
        self.env.close()

    def __del__(self):
        self.env.close()
