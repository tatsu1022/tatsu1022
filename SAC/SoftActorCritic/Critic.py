import torch
from torch.optim import Adam

from NeuralNetworkModel.BaseNetwork import BaseNetwork


class Critic(object):
    def __init__(self, num_inputs, num_actions, userDefinedSettings):
        self.q_network = TwinnedQNetwork(num_inputs, num_actions, hidden_units=userDefinedSettings.hidden_units,
                                         initializer=userDefinedSettings.initializer, lr=userDefinedSettings.lr).to(userDefinedSettings.DEVICE)
        self.target_network = TwinnedQNetwork(num_inputs, num_actions, hidden_units=userDefinedSettings.hidden_units,
                                              initializer=userDefinedSettings.initializer, lr=userDefinedSettings.lr).to(userDefinedSettings.DEVICE).eval()
        self.target_network.grad_false()
        self.soft_update_rate = userDefinedSettings.soft_update_rate

        # copy parameters of the learning network to the target network
        self.hard_update()  # only for initialization, not for learning

    def hard_update(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self):
        # sync params at soft_update_rate probability
        for target, q in zip(self.target_network.parameters(), self.q_network.parameters()):
            target.data.copy_(target.data * (1.0 - self.soft_update_rate) + q.data * self.soft_update_rate)

    def calc_loss(self, batch, policy, alpha, gamma_n):
        current_q1, current_q2 = self.q_network.calc_current_q(*batch)
        target_q = self.target_network.calc_target_q(*batch, policy, alpha, gamma_n)

        # We log means of Q to monitor training.
        mean_q1 = current_q1.detach().mean().item()
        mean_q2 = current_q2.detach().mean().item()

        # Critic loss_function is mean squared TD errors with priority weights.
        q1_loss = self.q_network.Q1.calc_loss(current_q1, target_q)
        q2_loss = self.q_network.Q2.calc_loss(current_q2, target_q)
        return q1_loss, q2_loss, mean_q1, mean_q2

    def update_params(self, q1_loss, q2_loss, grad_clip=None, retain_graph=False):
        self.q_network.Q1.update_params(q1_loss, grad_clip=None, retain_graph=False)
        self.q_network.Q2.update_params(q2_loss, grad_clip=None, retain_graph=False)


class TwinnedQNetwork(BaseNetwork):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer='xavier', lr=0.0003):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(num_inputs, num_actions, hidden_units, initializer, lr=lr)
        self.Q2 = QNetwork(num_inputs, num_actions, hidden_units, initializer, lr=lr)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x) # call for forward() ; nn.Module child
        q2 = self.Q2(x)
        return q1, q2

    def update_params_twinned_q(self, q1_loss, q2_loss, grad_clip=None, retain_graph=False):
        self.Q1.update_params(q1_loss, grad_clip=None, retain_graph=False)
        self.Q2.update_params(q2_loss, grad_clip=None, retain_graph=False)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        current_q1, current_q2 = self.forward(states, actions)
        return current_q1, current_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones, policy, alpha, gamma_n):
        with torch.no_grad():
            next_actions, next_entropies, _ = policy.sample(next_states)
            next_q1, next_q2 = self.forward(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + alpha * next_entropies   # Q(s',a') - alpha * log pi(a'|s')
        target_q = rewards + (1.0 - dones) * gamma_n * next_q   # r(s,a) + gamma * (Q(s',a') - alpha * log pi(a'|s'))
        return target_q


class QNetwork(BaseNetwork):
    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer='xavier', lr=0.0003):
        super(QNetwork, self).__init__()

        self.Q = self.create_linear_network(num_inputs + num_actions, 1, hidden_units=hidden_units, initializer=initializer)
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        q = self.Q(x)
        return q

    def calc_loss(self, current_q1, target_q):
        return self.loss_function(current_q1, target_q)
