import torch
from torch.optim import Adam
from torch.distributions import Normal

from NeuralNetworkModel.BaseNetwork import BaseNetwork


class Actor(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    epsilon = 1e-6

    def __init__(self, num_inputs, num_actions, userDefinedSettings):
        super(Actor, self).__init__()

        self.policy = self.create_linear_network(num_inputs, num_actions * 2, hidden_units=userDefinedSettings.hidden_units, initializer=userDefinedSettings.initializer)
        self.optimizer = Adam(self.policy.parameters(), lr=userDefinedSettings.lr)
        self.loss_function = torch.nn.MSELoss()
        self.DEVICE = userDefinedSettings.DEVICE # cuda or cpu
        self.to(self.DEVICE)
        self.squash_action_scale, self.squash_action_shift = self.calc_squash_parametes(userDefinedSettings.ACTION_MIN, userDefinedSettings.ACTION_MAX)

    def forward(self, states):
        mean, log_std = torch.chunk(self.policy(states), 2, dim=-1)  # torch.chunk(target, split_num, dim)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)  # cliping in the range(min, max)

        return mean, log_std

    def sample(self, states):
        # calculate Gaussian distribution of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        actions_not_squashed = normals.rsample()
        actions_squashed = self.squash_action(actions_not_squashed)
        tanh_2 = actions_squashed
        # calculate entropies
        log_probs = normals.log_prob(actions_not_squashed) - torch.log(1 - tanh_2.pow(2) + self.epsilon)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions_squashed, entropies, torch.tanh(means)

    def calc_loss(self, batch, critic, alpha):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = critic(states, sampled_action)
        q = torch.min(q1, q2)  # this gradient is used
        # Policy objective is maximization of (Q + alpha * entropy)
        loss = torch.mean((- q - alpha * entropy))
        return loss, entropy

    def calc_squash_parametes(self, action_min, action_max):
        squash_action_scale = torch.tensor((action_max - action_min) / 2.0).to(self.DEVICE) # mean of acton space
        squash_action_shift = torch.tensor(action_min + squash_action_scale).to(self.DEVICE)    # 
        return squash_action_scale, squash_action_shift

    def squash_action(self, actions_not_squashed):
        return self.squash_action_scale * torch.tanh(actions_not_squashed) + self.squash_action_shift
