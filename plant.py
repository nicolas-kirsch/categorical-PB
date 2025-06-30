import torch

class System(torch.nn.Module):
    def __init__(self, x0, horizon):
        self.x0 = x0
        self.horizon = horizon

    def step(self, x, u,d = None):
        """if d is not None:
            u = d[:, 0:+1, :] + u"""


        x = 0.95*x + u - d
        return x
    
    def rollout(self, controller, tau = 0.06,d = None, hard = False):
        x = d[:,0:1,:] if d is not None else self.x0.clone()
        xs = x.clone()
        us = torch.zeros(xs.shape[0],1,1)
        for t in range(1,self.horizon-1):

            u = controller(x,d[:,t:t+1,:],tau = tau,  hard = hard)
            x = self.step(x, u,d[:,t:t+1,:])

            xs = torch.cat((xs, x), 1)
            us = torch.cat((us, u), 1)
        return xs, us
