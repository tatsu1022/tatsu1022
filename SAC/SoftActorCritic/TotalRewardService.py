from collections import deque
import numpy as np


class TotalRewardService:

    def __init__(self, userDefinedSettings):
        self.n = userDefinedSettings.average_num_for_model_save
        self.train_total_reward_queue = deque(maxlen=self.n)
        self.test_total_reward_queue = deque(maxlen=self.n)

        self.train_max_average_total_reward = -999.9
        self.test_max_average_total_reward = -999.9

    def append_train(self, x):
        self.train_total_reward_queue.append(x)
    
    def append_test(self, total_reward):
        self.test_total_reward_queue.append(total_reward)

    def get_train_latest(self):
        return self.train_total_reward_queue[-1]

    def get_train_mean(self):
        return np.mean(self.train_total_reward_queue)

    def check_reward_peak(self, target='train', max_rate_threshold=0.9):
        if target == 'test':
            check_target_queue = self.test_total_reward_queue
            max_average_total_reward = self.test_max_average_total_reward
        elif target == 'train':
            check_target_queue = self.train_total_reward_queue
            max_average_total_reward = self.train_max_average_total_reward

        if np.average(check_target_queue) > max_average_total_reward:
            if target == 'test':
                self.test_max_average_total_reward = np.average(check_target_queue)
            if target == 'train':
                self.train_max_average_total_reward = np.average(check_target_queue)
            return True
        elif self.check_reward_threshold(check_target_queue, max_average_total_reward, max_rate_threshold):
            return True
        else:
            return False
    
    def check_reward_threshold(self, check_target_queue, max_average_total_reward, max_rate_threshold):
        target_reward = np.average(check_target_queue)

        if max_average_total_reward >= 0.:
            return target_reward > max_average_total_reward * max_rate_threshold
        else:
            return target_reward > max_average_total_reward * (2. - max_rate_threshold)

    def check_reward_peak_with_internal_parformance(self):
        if self.check_reward_peak(target='test') and self.check_reward_peak(target='train'):
            return True
