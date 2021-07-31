import time
import numpy as np
from numpy.core.overrides import array_function_dispatch

from ur5_mujoco.scripts.ur5_reduced_space_environment import UR5SpaceReducedEnvironment


class UR5SpaceReducedEnvironmentSweep(UR5SpaceReducedEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.distance_threshold = config.distance_threshold
        self.default_nframes = config.default_nframes
        self.stop_speed_threshold = config.stop_speed_threshold

        self.action_space = self.reduced_polar_action_space

    def step(self, action, nframes=None):
        if nframes is None: nframes = self.default_nframes

        current_state = self._get_obs()
        # print('action: {}'.format(action))
        self.set_reduced_polar_control_input(action)
        # self.set_joint3_controlled_value()
        self.sim.forward()

        for _ in range(nframes):
            self.sim.step()
        observation = self._get_obs()

        reward, done, info = self._get_reward(current_state, action, observation)

        return observation, reward, done, info
    
    def _get_reward(self, current_state, action, next_state):
        box_pos = self.sim.data.get_body_xpos('box')[:2]
        target_area_center = self.sim.data.get_site_xpos('target_area_geom')[:2]
        norm = np.linalg.norm(box_pos - target_area_center, axis=-1)
        
        ctrl_cost = self._control_cost(action)
        
        reward = - (norm + ctrl_cost)
        done = self._is_success()

        if done:    reward += 0.5

        info = {'norm': norm, 'control_cost': ctrl_cost}
        return reward, done, info

    def _control_cost(self, action):
        _control_cost_weight = 0.5
        control_cost = _control_cost_weight * np.linalg.norm(action)
        return control_cost

    def _is_success(self):    # whether the box is inside the area.
        box_vertices = self.get_box_vertices_pos()
        z_col = box_vertices[:,2]
        box_bottom_vertices = box_vertices[:,:2][np.argsort(z_col)][:4]
        target_area = self.get_target_area_vertices_pos()[:,:2]

        box_vel = self.sim.data.get_body_xvelp('box')[:2]   # only translational
        box_speed = np.linalg.norm(box_vel)
        
        x_min = min(target_area[:,0])
        x_max = max(target_area[:,0])
        y_min = min(target_area[:,1])
        y_max = max(target_area[:,1])

        vertices_in = []
        for vertex in box_bottom_vertices:
            x_in = x_min <= vertex[0] <= x_max
            y_in = y_min <= vertex[1] <= y_max
            vertex_in = x_in and y_in
            vertices_in.append(vertex_in)

        if np.all(vertices_in) and box_speed < self.stop_speed_threshold:
            return True
        else:
            return False


    def get_max_episode_steps(self):
        return self._max_episode_steps

    def random_action_sample(self):
        action = self.action_space.sample()
        return action