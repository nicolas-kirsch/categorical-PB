import torch

torch.manual_seed(2)

import numpy as np
from controller import Controller, Controller_Sigmoid
from plant import System

import matplotlib.pyplot as plt

x0 = torch.zeros((1,1,1))
x0[0,0,0] = 1
x_target = 10

horizon = 100
num_epochs = 5000
tau_0 = 1
tau = 0.05
sys = System(x0,horizon)
log_epochs = num_epochs // 10

controller = Controller()

optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)



best_loss = float('inf')
best_params = None

for epoch in range(num_epochs):
    optimizer.zero_grad()

    #tau = tau_0 * 1/np.log(0.7*epoch+2)

    # Simulate the system for 10 steps
    x_log, u_log = sys.rollout(controller, tau = tau)

    # Compute loss (mean squared error)
    target = torch.full_like(x_log,x_target)
    loss = torch.mean((x_log - target) ** 2 + 10*(u_log*(1-u_log)) )
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch}, Loss: {loss}')


        if loss < best_loss:
            best_loss = loss.item()
            best_params = controller.state_dict()


# Load the best parameters
controller.load_state_dict(best_params)


x_log, u_log = sys.rollout(controller, tau=tau)



x_log_hard, u_log_hard = sys.rollout(controller,tau=tau, hard=True)
loss = torch.mean((x_log_hard - target) ** 2 )
print(f'Final Loss: {loss}')

fig, ax = plt.subplots(2, 2)

ax[0,0].plot(x_log[0,:,:].detach().numpy(), label='State x')
ax[0,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[0,0].set_title('State x')
ax[0,0].legend()

ax[0,1].plot(u_log[0,:,:].detach().numpy(), label='Control u')
ax[0,1].set_title('Control u')
ax[0,1].legend()

ax[1,0].plot(x_log_hard[0,:,:].detach().numpy(), label='State x')
ax[1,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[1,0].set_title('State x')
ax[1,0].legend()

ax[1,1].plot(u_log_hard[0,:,:].detach().numpy(), label='Control u')
ax[1,1].set_title('Control u')
ax[1,1].legend()

plt.tight_layout()
plt.show()