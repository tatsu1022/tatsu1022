import torch
import torch.nn as nn


str_to_initializer = {
    'uniform': nn.init.uniform_,
    'normal': nn.init.normal_,
    'eye': nn.init.eye_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'he': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_
}


str_to_activation = {
    'elu': nn.ELU(),
    'hardshrink': nn.Hardshrink(),
    'hardtanh': nn.Hardtanh(),
    'leakyrelu': nn.LeakyReLU(),
    'logsigmoid': nn.LogSigmoid(),
    'prelu': nn.PReLU(),
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'rrelu': nn.RReLU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'logsoftmax': nn.LogSoftmax(),
    'softshrink': nn.Softshrink(),
    'softsign': nn.Softsign(),
    'tanh': nn.Tanh(),
    'tanhshrink': nn.Tanhshrink(),
    'softmin': nn.Softmin(),
    'softmax': nn.Softmax(dim=1),
    'none': None
}


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def create_linear_network(self, input_dim, output_dim, hidden_units=[],
                              hidden_activation='relu', output_activation=None,
                              initializer='xavier_uniform'):
        model = [] # define whole model construction - tf.sequential
        units = input_dim
        # hidden layers - dense
        for next_units in hidden_units:
            model.append(nn.Linear(units, next_units))  # append ful-connect
            model.append(str_to_activation[hidden_activation])  # append ReLU
            units = next_units
        
        # output layer
        model.append(nn.Linear(units, output_dim))
        if output_activation is not None:
            model.append(str_to_activation[output_activation])  # append Softmax etc.

        return nn.Sequential(*model).apply(self.initialize_weights(str_to_initializer[initializer]))  # append layers info to pytorch model

    def initialize_weights(self, initializer):
        def initialize(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                initializer(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        return initialize

    def update_params(self, loss, grad_clip=None, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if grad_clip is not None:
            for p in self.modules():
                torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
        self.optimizer.step()

    def grad_false(self):
        for param in self.parameters():
            param.requires_grad = False
