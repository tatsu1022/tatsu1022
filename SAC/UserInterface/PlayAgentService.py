import os
import torch

from SoftActorCritic.Actor import Actor
from .UserDefinedSettings import UserDefinedSettings
from Environment.EnvironmentFactory import EnvironmentFactory


class PlayAgentService(object):

    def run(self, learned_policy_head_path):
        userDefinedSettings = UserDefinedSettings()
        environmentFactory = EnvironmentFactory(userDefinedSettings)

        env = environmentFactory.generate()

        STATE_DIM, ACTION_DIM = env.get_state_action_space()
        policy = Actor(STATE_DIM[0], ACTION_DIM[0], userDefinedSettings)

        policy.load(os.path.join(learned_policy_head_path, 'model', 'policy.pth'))
        policy.grad_false()

        def exploit(state):
            state = torch.FloatTensor(state).unsqueeze(0).to(userDefinedSettings.DEVICE)
            with torch.no_grad():
                _, _, action = policy.sample(state)
            return action.cpu().numpy().reshape(-1)

        state = env.reset()
        done = False
        total_reward = 0.
        while not done:
            env.render()
            action = exploit(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            print('ACTION:', action)

        print('TOTAL REWARD:', total_reward)
