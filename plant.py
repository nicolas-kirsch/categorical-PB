import torch

class System(torch.nn.Module):
    def __init__(self, x0, horizon):
        self.x0 = x0
        self.horizon = horizon
          # Initialize heating state to 0.5 (OFF)

    def step(self, x, u_NN,u_bin, d = None, neural = True):
        """if d is not None:
            u = d[:, 0:+1, :] + u"""
        # Compute conditions
        on  = x < 8  # ON if too cold
        off = x > 12  # OFF if too hot

        # Apply bang-bang control with hysteresis
        self.heating = torch.where(on,  torch.ones_like(x),
                torch.where(off, torch.zeros_like(x),
                self.heating))
        
        
        # Else: keep the current state
        #self.binary = (1-(1-self.heating)*(1-u_bin))
        self.binary = u_bin.clone()
        #x = 0.95*x + u - d
        if neural:
            u = self.binary*(0.4)
        else:
            u = self.heating*0.4

        x = 0.99*x + u -d

        self.u = u.clone()
        return x
    
    def rollout(self, controller, tau = 0.06,d = None, hard = False, neural = True):
        self.heating = torch.full_like(d[:,0:1,:], 1)
        self.binary =  torch.full_like(d[:,0:1,:], 1)
        x = d[:,0:1,:] if d is not None else self.x0.clone()
        xs = x.clone()
        us = torch.zeros(xs.shape[0],1,1)
        self.u_out = torch.zeros(xs.shape[0],1,1)
        self.ubs = torch.zeros(xs.shape[0],1,1)


        self.hs = self.heating.clone()
        self.bs = self.binary.clone()   
        
        for t in range(1,self.horizon-1):

            u, u_bin = controller(x,d[:,t:t+1,:],tau = tau,  hard = hard)
            x = self.step(x, u,u_bin,d[:,t:t+1,:], neural=neural)

            xs = torch.cat((xs, x), 1)
            us = torch.cat((us, u), 1)
            self.u_out = torch.cat((self.u_out, self.u), 1)

            self.hs = torch.cat((self.hs, self.heating),1)
            self.bs = torch.cat((self.bs, self.binary), 1)
            self.ubs = torch.cat((self.ubs, u_bin), 1)
        return xs, us
