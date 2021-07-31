from SAC.UserInterface.UserDefinedSettings import UserDefinedSettings
from SAC.Environment.EnvironmentFactory import EnvironmentFactory

import time
import numpy as np


settings = UserDefinedSettings()
factory = EnvironmentFactory(settings)
model = factory.generate(settings.ENVIRONMENT_NAME)

def main(nstep=200, nframes=10, initialize=True):
    if initialize:  observation = model.reset()

    for _ in range(nstep):
        model.render()
        action = model.random_action_sample()
        observation, reward, done, info = model.step(action, nframes)
        '''
        print("=" * 10)
        print("action=",action)
        print("observation=",observation)
        print("reward=",reward)
        print("done=",done)
        print("info=",info)
        print()
        '''

def test_polar(nstep=120, nframes=10, initialize=True):
    if initialize:  observation = model.reset()

    #action = np.zeros(3*nstep).reshape(3, nstep)
    action_range = model.env.action_space.get_range()

    step1 = int(nstep/2)
    radius_sweep = np.linspace(action_range[0], action_range[1]-0.02, step1)
    # radius_max付近でtip_heightがconstant_tip_heightを超える。原因不明。
    radius_sweep = np.stack((np.zeros(step1), radius_sweep), axis=1)

    step2a = int(nstep/6)
    step2b = int(nstep/3)

    rot_sweep = np.concatenate((np.linspace(0, action_range[3], step2a), np.linspace(action_range[3], action_range[2], step2b)))
    rot_sweep = np.stack((rot_sweep, np.ones(step2a+step2b)*radius_sweep[-1,1]), axis=1)

    action_sweep = np.vstack((radius_sweep, rot_sweep))

    for i in range(nstep+20):
        model.render()
        try:
            action = action_sweep[i]
        except IndexError:
            action = [0, 0]
        observation, reward, done, info = model.step(action, nframes)
        '''
        print("=" * 10)
        print("action=",action)
        print("observation=",observation)
        print("reward=",reward)
        print("done=",done)
        print("info=",info)
        print()
        '''


if __name__ == '__main__':
    test_polar(100, 10)