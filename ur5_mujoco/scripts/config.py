import dataclasses
import numpy as np


# @dataclasses.dataclass(frozen=True)
@dataclasses.dataclass(frozen=False)
class Config:
    seed                : int  = 42
    xml_path            : str  = "/home/tatsu/internship/sac_workspace/ur5_mujoco/model/ur5_sweeping_v2_debug.xml"
    is_viewer_off_screen: bool = True
    is_render           : bool = True
    inner_step          : int  = 300
    max_episode_steps   : int  = 1000

    fix_gripper         : bool = True
    fps                 : int  = 30

    distance_threshold  : float = 0.03
    stop_speed_threshold: float = 0.03
    default_nframes     : int   = 5
    constant_tip_height : float = 0.15

    viewer_x_coordinate : float = 0
    viewer_y_coordinate : float = 0
    viewer_z_coordinate : float = 0.15
    viewer_elevation    : float = -45
    viewer_azimuth      : float = 180
    viewer_distance_rate: float = 0.

    width_capture  : int = 64 * 7
    height_capture : int = 64 * 7

    save_video : bool = False


    def viewer_param(self):
        param = {}
        param["viewer_x_coordinate"]  = self.viewer_x_coordinate
        param["viewer_y_coordinate"]  = self.viewer_y_coordinate
        param["viewer_z_coordinate"]  = self.viewer_z_coordinate
        param["viewer_elevation"]     = self.viewer_elevation
        param["viewer_azimuth"]       = self.viewer_azimuth
        param["viewer_distance_rate"] = self.viewer_distance_rate
        return param


if __name__ == '__main__':
    config = Config()
    print(config)
    print(config.viewer_param())
