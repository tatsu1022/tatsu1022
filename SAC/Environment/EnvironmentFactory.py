
class EnvironmentFactory():
    def __init__(self, userDefinedSettings):
        self.ENVIRONMENT_NAME = userDefinedSettings.ENVIRONMENT_NAME
        self.userDefinedSettings = userDefinedSettings

    def generate(self, env_name=None):
        if env_name is not None:    self.ENVIRONMENT_NAME = env_name
        if self.ENVIRONMENT_NAME == 'HalfCheetah':
            from .HalfCheetah import HalfCheetah
            return HalfCheetah(self.userDefinedSettings)
        if self.ENVIRONMENT_NAME == 'SwingUp':
            from .SwingUp import SwingUp
            return SwingUp(self.userDefinedSettings)
        #if self.ENVIRONMENT_NAME == 'Excavator':
        #    from .Excavator.Excavator import Excavator
        #    return Excavator(self.userDefinedSettings)
        if self.ENVIRONMENT_NAME == 'Pendulum':
            from .Pendulum import Pendulum
            return Pendulum(self.userDefinedSettings)
        
        if self.ENVIRONMENT_NAME == 'RobotPush':
            from .RobotPush import RobotPush
            return RobotPush(self.userDefinedSettings)

        if self.ENVIRONMENT_NAME == 'UR5':
            from .UR5.UR5 import UR5
            return UR5(self.userDefinedSettings)
        if self.ENVIRONMENT_NAME == 'UR5Reach':
            from .UR5.UR5_reach import UR5Reach
            return UR5Reach(self.userDefinedSettings)
        if self.ENVIRONMENT_NAME == 'UR5Sweep':
            from .UR5.UR5_sweep import UR5Sweep
            return UR5Sweep(self.userDefinedSettings)
        if self.ENVIRONMENT_NAME == 'UR5ReducedSpace':
            from .UR5.UR5_reduced_space import UR5ReducedSpace
            return UR5ReducedSpace(self.userDefinedSettings)
