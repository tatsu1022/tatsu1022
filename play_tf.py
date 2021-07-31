import os
import tensorflow as tf

from sac_tf import Actor
from SAC.UserInterface.UserDefinedSettings import UserDefinedSettings
from SAC.Environment.EnvironmentFactory import EnvironmentFactory


class PlayAgentService(object):

    def run(self, learned_policy_head_path):
        userDefinedSettings = UserDefinedSettings()
        environmentFactory = EnvironmentFactory(userDefinedSettings)

        env = environmentFactory.generate()

        STATE_DIM, ACTION_DIM = env.get_state_action_space()

        policy = tf.keras.models.load_model(os.path.join(learned_policy_head_path, 'model', 'policy'))

        def exploit(state):
            state = tf.expand_dims(tf.constant(state, dtype=tf.float64), axis=0)
            _, _, action = policy(tf.stop_gradient(state))
            return action.numpy().reshape(-1)

        state = env.reset()
        done = False
        total_reward = 0.
        step = 0
        while not done and step < 1000:
            step += 1
            env.render()
            action = exploit(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            print(  'STATE: {}\n'.format(next_state), 
                    'ACTION: {}\n'.format(action), 
                    'REWARD: {}\n'.format(reward), 
                    'DONE: {}\n'.format(done), 
                    'INFO: {}\n'.format(info)
                )
            # print('box pos: {}'.format(env.env.sim.data.get_body_xpos('box_mass_center')))

        print('TOTAL REWARD:', total_reward)


if __name__ == '__main__':
    model_path = '/home/tatsu/internship/sac_workspace/logs/ForUse/'
    playAgentService = PlayAgentService()
    playAgentService.run(model_path)