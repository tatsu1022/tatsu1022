import torch
from torch.optim import Adam

from NeuralNetworkModel.BaseNetwork import BaseNetwork


class EntropyTerm(BaseNetwork):

    def __init__(self, ACTION_DIM, userDefinedSettings):
        self.entropy_tuning = userDefinedSettings.entropy_tuning

        if self.entropy_tuning:
            # Target entropy is -|A|. hyper parameter: -1 x ActionDim is recommended
            target_entropy = ACTION_DIM[0] * userDefinedSettings.entropy_tuning_scale
            target_entropy = (target_entropy, )
            self.target_entropy = -torch.prod(torch.Tensor(target_entropy).to(userDefinedSettings.DEVICE)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=userDefinedSettings.DEVICE)
            self.alpha = self.log_alpha.exp()
            self.optimizer = Adam([self.log_alpha], lr=userDefinedSettings.lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(userDefinedSettings.entropy_coefficient).to(userDefinedSettings.DEVICE)

    def calc_entropy_loss(self, entropy):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach())
        return entropy_loss

    def update_alpha(self, entropies):
        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies)
            self.update_params(entropy_loss)
            self.alpha = self.log_alpha.exp()
            return entropy_loss
        else:
            return None
