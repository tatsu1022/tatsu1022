import time
import numpy as np
from numpy.core.overrides import array_function_dispatch

from transforms3d.euler import euler2quat, quat2euler
from transforms3d.derivations.quaternions import quat2mat

from ur5_mujoco.scripts.ur5_reduced_space_environment import UR5SpaceReducedEnvironment


class UR5SpaceReducedEnvironmentSweep(UR5SpaceReducedEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.distance_threshold = config.distance_threshold
        self.default_nframes = config.default_nframes
        self.stop_speed_threshold = config.stop_speed_threshold

        self.action_space = self.reduced_polar_action_space

        self.target_area_vertices = self.get_target_area_vertices_pos()
        self.target_area_range = self.target_area_range()       # must call after ramdomize

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
        box_area_norm = np.linalg.norm(box_pos - target_area_center, axis=-1)
        
        ctrl_cost = self._control_cost(action)

        done = self._is_success()
        done_reward = 0.5
        reward = - (box_area_norm + ctrl_cost) + done_reward * done

        info = {'box_area_norm': box_area_norm, 'control_cost': ctrl_cost}
        return reward, done, info

    def _control_cost(self, action):
        _control_cost_weight = 0.5
        control_cost = _control_cost_weight * np.linalg.norm(action)
        return control_cost

    def _is_success(self):    # num of distributed points inside the area
        box_vertices = self.get_box_vertices_pos()
        z_col = box_vertices[:,2]
        box_bottom_vertices = box_vertices[:,:2][np.argsort(z_col)][:4]
        
        x_min, x_max, y_min, y_max = self.target_area_range
        vertices_in = []
        for vertex in box_bottom_vertices:  # assuming only translational ramdomization for target area 
            x_in = x_min <= vertex[0] <= x_max
            y_in = y_min <= vertex[1] <= y_max
            vertex_in = x_in and y_in
            vertices_in.append(vertex_in)

        box_vel = self.sim.data.get_body_xvelp('box')[:2]   # only translational
        box_speed = np.linalg.norm(box_vel)

        if np.all(vertices_in) and box_speed < self.stop_speed_threshold:
            return True
        else:
            return False

    def get_box_vertices_pos(self):
        box_pos = self.sim.data.get_body_xpos('box')
        quat = self.sim.data.get_body_xquat('box')
        rotmat = np.array(quat2mat(quat)).astype(np.float32)
        tfmat = np.vstack((np.hstack((rotmat, box_pos.reshape(-1,1))), [0,0,0,1]))
        
        box_dimension = self.box_dimension  # dimension of box(cuboid): [x,y,z] <- randomization
        vertices_pos = []
        for b in range(8):  # cuboid: 8
            mask = format(b, '#05b')[2:]    # flg for * -1
            vertex = np.array([-box_dimension[i] if int(mask[i]) else box_dimension[i] 
                                for i in range(len(box_dimension))]).astype(np.float32) # no need to box_dimension/2
            vertex = np.matmul(tfmat, np.concatenate((vertex, [1])))
            assert vertex[3] == 1.
            vertices_pos.append(vertex[:3])

        return np.array(vertices_pos)

    def get_target_area_vertices_pos(self): # not considering the rotation
        area_pos = self.sim.data.get_site_xpos('target_area_geom')
        target_area_dimension = self.target_area_dimension[:2] # dimension of target area, z_pos is fixed to 0.
        
        vertices_pos = []
        for b in range(4):  # square: 4
            mask = format(b, '#04b')[2:]
            vertex = np.array([-target_area_dimension[i] if int(mask[i]) else target_area_dimension[i] 
                                for i in range(len(target_area_dimension))]).astype(np.float32)
            vertex += area_pos[:2]
            vertex = np.concatenate((vertex, [0]))
            vertices_pos.append(vertex)
        return np.array(vertices_pos)
    
    def get_target_area_range(self):
        x_min = min(self.target_area_vertices[:,0])
        x_max = max(self.target_area_vertices[:,0])
        y_min = min(self.target_area_vertices[:,1])
        y_max = max(self.target_area_vertices[:,1])
        return np.array([x_min, x_max, y_min, y_max])

    def get_max_episode_steps(self):
        return self._max_episode_steps

    def random_action_sample(self):
        action = self.action_space.sample()
        return action