import gym
# import gym_cartpole_swingup  # noqa
import time

from .InterfaceEnvironment import InterfaceEnvironment


class SwingUp(InterfaceEnvironment):
    def __init__(self, userDefinedSettings):
        # Could be one of:
        # CartPoleSwingUp-v0, CartPoleSwingUp-v1
        # If you have PyTorch installed:
        # TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
        self.env = gym.make("CartPole-v1")
        self.RENDER_INTERVAL = userDefinedSettings.RENDER_INTERVAL

    def get_state_action_space(self):
        STATE_DIM = self.env.observation_space.shape
        ACTION_DIM = self.env.action_space.shape
        return STATE_DIM, ACTION_DIM

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def get_max_episode_steps(self):
        return self.env._max_episode_steps

    def random_action_sample(self):
        action = self.env.action_space.sample()
        return action

    def render(self):
        self.env.render()
        time.sleep(self.RENDER_INTERVAL)

    def __del__(self):
        self.env.close()
