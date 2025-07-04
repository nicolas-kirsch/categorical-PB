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

class Controller_range_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1), 
             # output logit
        )

        self.net_range = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # output logit
        )

        

    def forward(self, x, tau=1.0, hard=False):

        out_range = self.net_range(x)
        out_range = 2 * out_range  # Scale to [0.5, 5.0]
        
        output = self.net(x)  # shape: [batch, 1]

        y_soft = torch.sigmoid(tau* output)

        if hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-through estimator
            return out_range
        #*((y_hard - y_soft).detach() + y_soft)
        else:
            return out_range
        #*y_soft
                       


class Controller_range(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1), 
             # output logit
        )

        self.net_range = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # output logit
        )
        

    def forward(self, x,d, tau=1.0, hard=False):
        input = torch.cat((x,d), dim=2)

        out_range = self.net_range(input)
        out_range = -0.4 + 2 * out_range  # Scale to [0.5, 5.0]

        logits = self.net(input)  # shape: [batch, 1]
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

        if hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-through estimator
            binary_out = (y_hard - y_soft).detach() + y_soft
            return out_range, binary_out
        #*((y_hard - y_soft).detach() + y_soft)
        else:
            return out_range, y_soft
        #*y_soft
