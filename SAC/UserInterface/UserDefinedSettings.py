import os
import warnings
import datetime

warnings.simplefilter('ignore', FutureWarning)  # noqa


class UserDefinedSettings(object):

    def __init__(self):
        self.DEVICE = 'cuda'  # cuda, cpu
        self.ENVIRONMENT_NAME = 'UR5ReducedSpace'  # HalfCheetah, Excavator, Pendulum
        self.RENDER_INTERVAL = 0.05  # [s]
        current_time = datetime.datetime.now()
        file_name = 'M{:0=2}D{:0=2}H{:0=2}M{:0=2}'.format(current_time.month,current_time.day,current_time.hour,current_time.minute)
        self.LOG_DIRECTORY = os.path.join('logs', self.ENVIRONMENT_NAME, 'sac', file_name)

        self.num_steps = 75000 # 1e6
        self.batch_size = 256
        self.learning_rate = 0.0003
        self.hidden_units = [256, 256]
        self.memory_size = 1e6
        self.gamma = 0.99
        self.soft_update_rate = 0.005
        self.entropy_tuning = True
        self.entropy_tuning_scale = 1.0
        self.entropy_coefficient = 0.2
        self.multi_step_reward_num = 1
        self.grad_clip = None
        self.updates_per_step = 1
        self.start_steps = 10000
        self.target_update_interval = 1 # episode num
        self.evaluate_interval = 5 # episode num
        self.initializer = 'xavier'
        self.ACTION_MIN = -1.0
        self.ACTION_MAX = 1.0
        self.EPISODE_LENGTH = None
        self.run_num_per_evaluate = 1
        self.average_num_for_model_save = 5
        self.learning_episode_num = 1000000
"""      
    steps = 0
    episodes = 0

    for _ in range(learning_episode_num):   1000000
        episodes += 1
        env.reset()
        episode_steps = 0

        while not done:         1 episode
            env.step(action)
            steps += 1 
            episode_steps += 1

            if len(memory) > batch_size and steps >= start_steps:
                for _ in range(updates_per_step):
                    learn() -> update Q-functions, Policy, alpha


        if episodes % evaluate_interval == 0:
            evaluate() 

        if steps > num_steps:     episodes * steps > learning_episode_num
            break           
"""