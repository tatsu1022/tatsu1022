import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from .Actor import Actor
from .Critic import Critic
from .EntropyTerm import EntropyTerm
from DatasetHandler.MultiStepReplayMemory import MultiStepMemory
from .TotalRewardService import TotalRewardService


class SACAgent(object):

    def __init__(self, env, userDefinedSettings):
        self.num_steps = userDefinedSettings.num_steps
        self.soft_update_rate = userDefinedSettings.soft_update_rate
        self.batch_size = userDefinedSettings.batch_size
        self.start_steps = userDefinedSettings.start_steps
        self.gamma_n = userDefinedSettings.gamma ** userDefinedSettings.multi_step_reward_num
        self.entropy_tuning = userDefinedSettings.entropy_tuning
        self.grad_clip = userDefinedSettings.grad_clip
        self.updates_per_step = userDefinedSettings.updates_per_step
        # self.log_interval = userDefinedSettings.log_interval
        self.target_update_interval = userDefinedSettings.target_update_interval
        self.evaluate_interval = userDefinedSettings.evaluate_interval
        self.DEVICE = userDefinedSettings.DEVICE
        self.run_num_per_evaluate = userDefinedSettings.run_num_per_evaluate
        self.env = env
        self.learning_episode_num = userDefinedSettings.learning_episode_num

        self.STATE_DIM, self.ACTION_DIM = self.env.get_state_action_space() # |S| and |A|

        self.actor = Actor(self.STATE_DIM[0], self.ACTION_DIM[0], userDefinedSettings)
        self.critic = Critic(self.STATE_DIM[0], self.ACTION_DIM[0], userDefinedSettings)
        self.entropyTerm = EntropyTerm(self.ACTION_DIM, userDefinedSettings)    # alpha

        self.memory = MultiStepMemory(self.STATE_DIM, self.ACTION_DIM, userDefinedSettings) # replay pool

        self.log_dir = userDefinedSettings.LOG_DIRECTORY
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.summary_dir = os.path.join(self.log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.totalRewardService = TotalRewardService(userDefinedSettings)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0

    def run(self):
        # iterate train_episode for episode num times.
        for _ in range(self.learning_episode_num):
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def choose_action(self, state):
        if self.start_steps > self.steps:
            action = self.env.random_action_sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # random action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # mean action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            _, _, action = self.actor.sample(state)
        return action.cpu().numpy().reshape(-1)

    def train_episode(self):
        self.episodes += 1
        total_reward = 0.
        episode_steps = 0
        state = self.env.reset()
        done = False

        while not done:
            action = self.choose_action(state)  # a~pi(a|s)
            next_state, reward, done, _ = self.env.step(action) # s'~p(s'|s,a)
            self.steps += 1
            episode_steps += 1
            total_reward += reward

            self.memory.append(state, action, reward, next_state, done)

            if self.is_update():    # |memory| > |batch| and steps >= start_steps
                for _ in range(self.updates_per_step):
                    self.learn()

            state = next_state 

            # print('Episodes:{}, Steps:{}, Reward:{}'.format(self.episodes,episode_steps,reward))

        # We log running mean of training rewards.
        print('Total Reward:{}'.format(total_reward))
        self.totalRewardService.append_train(total_reward)

        if self.episodes % self.evaluate_interval == 0:
            self.evaluate()

        if self.decide_model_save():
            print('### model updated!!')
            self.save_models()
        else:
            print('### model keep')

        train_total_reward = self.totalRewardService.get_train_latest()
        self.writer.add_scalar('reward/train', train_total_reward, self.steps)

    def decide_model_save(self):
        return self.totalRewardService.check_reward_peak()
        
    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            self.critic.soft_update()

        batch = self.memory.sample(self.batch_size)

        q1_loss, q2_loss, mean_q1, mean_q2 = self.critic.calc_loss(batch, self.actor, self.entropyTerm.alpha, self.gamma_n)
        policy_loss, entropies = self.actor.calc_loss(batch, self.critic.q_network, self.entropyTerm.alpha)

        self.actor.update_params(policy_loss, self.grad_clip)  # calculated at 1
        self.critic.update_params(q1_loss, q2_loss, self.grad_clip)  # 2 or 3
        entropy_loss = self.entropyTerm.update_alpha(entropies)  # 2 or 3

        if entropy_loss is not None:
            self.writer.add_scalar('loss/alpha', entropy_loss.detach().item(), self.steps)
        self.writer.add_scalar('loss/Q1', q1_loss.detach().item(), self.learning_steps)
        self.writer.add_scalar('loss/Q2', q2_loss.detach().item(), self.learning_steps)
        self.writer.add_scalar('loss/policy', policy_loss.detach().item(), self.learning_steps)
        self.writer.add_scalar('stats/alpha', self.entropyTerm.alpha.detach().item(), self.learning_steps)
        self.writer.add_scalar('stats/mean_Q1', mean_q1, self.learning_steps)
        self.writer.add_scalar('stats/mean_Q2', mean_q2, self.learning_steps)
        self.writer.add_scalar('stats/entropy', entropies.detach().mean().item(), self.learning_steps)

    def evaluate(self):
        episodes = self.run_num_per_evaluate
        total_rewards = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0.
            done = False
            steps = 0
            while not done:
                steps += 1
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                print('Test, Episodes:{}, Steps:{}, Reward:{}'.format(self.episodes ,steps, reward))
            total_rewards[i] = total_reward

        mean_total_reward = np.mean(total_rewards)

        self.writer.add_scalar(
            'reward/test', mean_total_reward, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'reward: {mean_total_reward:<5.1f}')
        print('-' * 60)

        self.totalRewardService.append_test(mean_total_reward)

    def save_models(self):
        self.actor.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.q_network.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic.target_network.save(os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.writer.close()
        self.env.close()
