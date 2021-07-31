import time
import numpy as np

from ur5_mujoco.scripts.ur5_simulation_environment import UR5SimulationEnvironment


class UR5SweepingEnvironment(UR5SimulationEnvironment):
    def __init__(self, config):
        config.xml_path = "/home/tatsu/internship/sac_workspace/ur5_mujoco/model/ur5_sweeping.xml"
        super().__init__(config)
        self.distance_threshold = config.distance_threshold
        self.default_nframes = config.default_nframes

    def step(self, action, nframes=None):
        if nframes is None: nframes = self.default_nframes

        current_state = self._get_obs()
        self.set_difference_control_input(action)
        for _ in range(nframes):
            self.sim.step()
        observation = self._get_obs()

        reward, done, info = self._get_reward(current_state, action, observation)

        time.sleep(.0001)   # prevents mujoco_py.builder.MujocoException: Unknown warning type Time
        return observation, reward, done, info
    
    def _get_reward(self, current_state, action, next_state):
        tcp = self.sim.data.get_body_xpos('basket_center')
        targ = self.sim.data.get_body_xpos('box_mass_center')
        norm = np.linalg.norm(tcp - targ, axis=-1)
        
        ctrl_cost = self._control_cost(action)
        
        reward = - (norm + ctrl_cost)
        done = self._is_success(norm)

        if done:    reward += 0.5

        info = {'norm': norm, 'control_cost': ctrl_cost}
        return reward, done, info

    def _control_cost(self, action):
        _control_cost_weight = 0.5
        control_cost = _control_cost_weight * np.linalg.norm(action)
        return control_cost

    def _is_success(self, norm):
        return norm < self.distance_threshold

    def get_max_episode_steps(self):
        return self._max_episode_steps

    def random_action_sample(self):
        action = self.action_space.sample()
        return action