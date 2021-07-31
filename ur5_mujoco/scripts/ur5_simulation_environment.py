import copy
from collections import OrderedDict

from numpy.core.numerictypes import obj2sctype
import cv2
import numpy as np
import mujoco_py
from mujoco_py.modder import TextureModder, CameraModder
import gym
from transforms3d.euler import euler2quat, quat2euler
# from model_geometry_names_handler import ModelGeometryNamesHandler


class UR5SimulationEnvironment:
    def __init__(self, config):
        self.seed = config.seed
        self.xml_path = config.xml_path
        self.model = mujoco_py.load_model_from_path(config.xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.texture_modder = TextureModder(self.sim)
        self.camera_modder = CameraModder(self.sim)
        self.is_viewer_off_screen = config.is_viewer_off_screen
        self.is_render = config.is_render
        self.viewer = self.select_viewer()
        self.width_capture = config.width_capture
        self.height_capture = config.height_capture
        self.viewer_position = config.viewer_param()
        self.inner_step = config.inner_step
        self._max_episode_steps = config.max_episode_steps

        self.fix_gripper = config.fix_gripper
        self.set_random_seed()
        self.windowName = "ur5"

        self.save_video = config.save_video
        self.fps = config.fps
        if self.save_video:
            self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.cam_videos = {}
            for camera_name in self.sim.model.camera_names:
                self.cam_videos[camera_name] = cv2.VideoWriter(camera_name + '.mp4', self.fourcc, self.fps, (self.width_capture, self.height_capture))

            self.video = cv2.VideoWriter('video.mp4', self.fourcc, self.fps, (self.width_capture*3, self.height_capture))

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy() 

        self._set_action_space()
        self._set_observation_space()

        return None


    def _set_action_space(self):    # 6 DoF for wrists & 2 DoF for gripper
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        if self.fix_gripper:    bounds = bounds[:6]        
        low, high = bounds.T
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self):
        self.observation_space = self._convert_observation_to_space(self._get_obs())
        return self.observation_space

    def _convert_observation_to_space(self, observation):
        if isinstance(observation, dict):
            space = gym.spaces.Dict(OrderedDict([
                (key, self._convert_observation_to_space(value))
                for key, value in observation.items()
            ]))
        elif isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -float('inf'), dtype=np.float32)
            high = np.full(observation.shape, float('inf'), dtype=np.float32)
            space = gym.spaces.Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)

        return space

    def select_viewer(self):
        if self.is_viewer_off_screen:
            return mujoco_py.MjRenderContextOffscreen(self.sim, 0)
        else:
            return mujoco_py.MjViewer(self.sim)

    def set_random_seed(self):
        np.random.seed(self.seed)

    def _render_and_convert_color(self, camera_name):
        img = self.sim.render(width=self.width_capture, height=self.height_capture, camera_name=camera_name, depth=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[::-1]
        return img

    def render(self, render_num=1):
        for index_render_num in range(render_num):
            img = {}
            for camera_name in self.sim.model.camera_names:
                img[camera_name] = self._render_and_convert_color(camera_name=camera_name)
                if self.save_video: self.cam_videos[camera_name].write(img[camera_name])
            self.render_show(img)
        return img

    def render_show(self, img):
        if self.is_render == 1:
            cv2.imshow(self.windowName, np.hstack(img.values()))
            if self.save_video: self.video.write(np.hstack(img.values()))
            cv2.waitKey(20)

    def viewer_render(self):
        self.viewer.render()

    def check_camera_pos(self):
        self.sim.reset()
        for i in range(100):
            diff = 0.1
            self.model.cam_pos[0][0] = self.rs.uniform(self.default_cam_pos[0][0] - diff, self.default_cam_pos[0][0] + diff)  # x-axis
            self.model.cam_pos[0][1] = self.rs.uniform(self.default_cam_pos[0][1] - diff, self.default_cam_pos[0][1] + diff)  # x-axis
            self.model.cam_pos[0][2] = self.rs.uniform(self.default_cam_pos[0][2] - diff, self.default_cam_pos[0][2] + diff)  # z-axis
            self.sim.step()
            self.render()

    def set_camera_position(self):
        self.viewer.cam.lookat[0] = self.camera_position["x_coordinate"]
        self.viewer.cam.lookat[1] = self.camera_position["y_coordinate"]
        self.viewer.cam.lookat[2] = self.camera_position["z_coordinate"]
        self.viewer.cam.elevation = self.camera_position["elevation"]   # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = self.camera_position["azimuth"]     # camera rotation around the camera's vertical axis
        self.viewer.cam.distance = self.model.stat.extent * self.camera_position["distance_rate"]

    def get_state(self, is_only_qpos_qpos):
        env_state = copy.deepcopy(self.sim.get_state())
        if is_only_qpos_qpos:
            env_state = {"qpos": env_state.qpos, "qvel": env_state.qvel}
        return env_state

    def set_state(self, qpos, qvel):
        if qpos is None:    qpos = np.zeros(self.model.nq)
        if qvel is None:    qvel = np.zeros(self.model.nv)
        if not (qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)):
            print("\n***************************************")
            print(" model.nq = {} | qpos.nq = {}".format(self.model.nq, qpos.shape[0]))
            print(" model.nv = {} | qvel.nv = {}".format(self.model.nv, qvel.shape[0]))
            print("\n   dimension is incorrect ")
            print("\n***************************************")
            raise AssertionError("^^^^^^^^^")
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)

        self.sim.set_state(new_state)
        self.sim.data.ctrl[:8] = qpos[:8] 
        if self.fix_gripper: self.sim.data.ctrl[6:8] = self.init_qpos[6:8]
        self.sim.forward()

    def reset_env(self, qpos_init=None, qvel_init=None, target_position_init=None):
        self.sim.reset()
        # import ipdb
        # ipdb.set_trace()
        # self.set_jnt_range()
        # self.set_target_position(target_position_init)
        if qpos_init is not None:   self.init_qpos = qpos_init
        if qvel_init is not None:   self.init_qvel = qvel_init
        self.set_state(self.init_qpos, self.init_qvel)

        observation = self._get_obs()
        # self.render()
        return observation
    
    def _get_obs(self):
        qpos = self.sim.data.qpos.flat.copy()
        qvel = self.sim.data.qvel.flat.copy()
        observation = np.concatenate((qpos, qvel)).ravel()
        return observation

    def set_target_position(self, target_position):
        self._target_position = target_position

    # def set_jnt_range(self):
    #     # --- claw ---
    #     for jnt_index in range(9):
    #         self.sim.model.jnt_range[jnt_index, 0] = self.claw_jnt_range_lb[jnt_index]
    #         self.sim.model.jnt_range[jnt_index, 1] = self.claw_jnt_range_ub[jnt_index]

    #     # --- valve ---
    #     self.sim.model.jnt_range[self._valve_jnt_id, 0] = self.valve_jnt_range_lb
    #     self.sim.model.jnt_range[self._valve_jnt_id, 1] = self.valve_jnt_range_ub

    def step(self):
        self.sim.step()

    def step_multi(self, inner_step=-1):
        assert type(inner_step) == int
        if inner_step < 0:
            inner_step = self.inner_step
        for i in range(inner_step):
            self.sim.step()

    def set_control_input(self, ctrl):
        ctrl_dim = 6 if self.fix_gripper else 8
        assert ctrl.shape[0] == ctrl_dim
        self.sim.data.ctrl[:ctrl_dim] = ctrl
        self.sim.forward()

    def set_difference_control_input(self, diff_ctrl):
        ctrl_dim = 6 if self.fix_gripper else 8
        assert diff_ctrl.shape[0] == ctrl_dim
        self.sim.data.ctrl[:ctrl_dim] = self.sim.data.qpos[:ctrl_dim] + diff_ctrl
        self.sim.forward()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

        if self.save_video:
            for camera_name in self.sim.model.camera_names:
                self.cam_videos[camera_name].release()
            self.video.release()       
