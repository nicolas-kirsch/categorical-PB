import torch

class System(torch.nn.Module):
    def __init__(self, x0, horizon):
        self.x0 = x0
        self.horizon = horizon

    def step(self, x, u):
        x = 0.92 * x + u
        return x
    
    def rollout(self, controller, tau = 0.06, hard = False):
        x = self.x0
        xs = x.clone()
        us = torch.zeros(xs.shape[0],1,1)
        for _ in range(self.horizon):
            u = controller(x,tau = tau, hard = hard)
            x = self.step(x, u)

            xs = torch.cat((xs, x), 1)
            us = torch.cat((us, u), 1)
        return xs, us
