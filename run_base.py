import torch

torch.manual_seed(2)

import numpy as np
from controller import Controller, Controller_Sigmoid, Controller_range, Controller_range_sigmoid
from plant import System
from data import Dataset
import matplotlib.pyplot as plt

nb = 400

x0 = torch.zeros((nb,1,1))
#x0[0,0,0] = 10
x_target = 10

horizon = 200
num_epochs = 5000

tau_0 = 1
tau = 0.005
sys = System(x0,horizon)
log_epochs = num_epochs // 10

controller = Controller_range()
data = Dataset(x0, horizon)
d = data.generate_data()
test_data = data.generate_data()
test_data = test_data[:3, :, :]  # Use only the first time step for testing
test_data[0,0,:] = 0 
test_data[2,0,:] = 6 
test_data[1,0,:] = 3 

test_data[0,1:,:] = 0.3
test_data[2,1:,:] = 0.9
test_data[1,1:,:] = 0
dist = [0.3,0.9,1.8]

target = 10

x_log, u_log = sys.rollout(controller, tau = tau,d=test_data, neural = False)
loss = torch.mean((x_log - target) ** 2 )
print(loss)

plt.plot(x_log[1,:,0].detach().numpy(), label='x0')
plt.show()