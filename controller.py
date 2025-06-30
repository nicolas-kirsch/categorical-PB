import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # output logit
        )

    def forward(self, x, tau=1.0, hard=False):
        logits = self.net(x)  # shape: [batch, 1]
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

        if hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-through estimator
            return (y_hard - y_soft).detach() + y_soft
        else:
            return y_soft


class Controller_Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # output logit
        )

    def forward(self, x, tau=1.0, hard=False):

        output = self.net(x)  # shape: [batch, 1]

        y_soft = torch.sigmoid(tau* output)

        if hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-through estimator
            return (y_hard - y_soft).detach() + y_soft
        else:
            return y_soft
