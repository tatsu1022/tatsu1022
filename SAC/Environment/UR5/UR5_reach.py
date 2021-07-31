"""
インターンで使うUR5シミュレーション環境のテンプレート．
各関数はこのまま使えばよい．
関数の中身はmujoco環境に合うように書き換える必要あり．
"""

import sys, pathlib, time
parent_dir = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())
sys.path.append(parent_dir)

import numpy as np

from SAC.Environment.InterfaceEnvironment import InterfaceEnvironment

from ur5_mujoco.scripts.ur5_reaching_environment import UR5ReachingEnvironment
from ur5_mujoco.scripts.config import Config


class UR5Reach(InterfaceEnvironment):
    def __init__(self, userDefinedSettings):
        config = Config()
        self.env = UR5ReachingEnvironment(config)
        self.RENDER_INTERVAL = userDefinedSettings.RENDER_INTERVAL

    def get_state_action_space(self):
        STATE_DIM = self.env.observation_space.shape
        ACTION_DIM = self.env.action_space.shape
        return STATE_DIM, ACTION_DIM

    def reset(self):
        state = self.env.reset_env()
        return state

    def step(self, action, nframes=None):
        next_state, reward, done, info = self.env.step(action, nframes)
        return next_state, reward, done, info

    def get_max_episode_steps(self):
        return self.env._max_episode_steps

    def random_action_sample(self):
        action = self.env.action_space.sample()
        return action  

    def render(self):
        self.env.render()
        # time.sleep(self.RENDER_INTERVAL)
    
    def close(self):
        self.env.close()

    def __del__(self):
        self.env.close()