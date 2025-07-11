import torch
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, x0, horizon):
        self.x0 = x0
        self.horizon = horizon
        self.batch_size = self.x0.shape[0]

    def generate_data(self):
        d = torch.zeros((self.batch_size, self.horizon, 1))
        d = torch.zeros((self.batch_size, self.horizon, 1))
        d[:, 0, :] = 5.0 * torch.rand((self.batch_size, 1))


        # Sample one uniform value per batch in [0.3, 1.0]

        #u = 0.9 + 1.1*torch.rand((self.batch_size, 1, 1))   
        u = torch.rand((self.batch_size, 1, 1))   
        d[:, 1:70, :] = -u  # Fill from time step 70 onward (inclusive)
         
        return d


data = Dataset(torch.zeros((1, 1, 1)), 100)

