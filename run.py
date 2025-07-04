import torch

torch.manual_seed(2)

import numpy as np
from controller import Controller, Controller_Sigmoid, Controller_range, Controller_range_sigmoid
from plant import System
from data import Dataset
import matplotlib.pyplot as plt
from collections import OrderedDict
nb = 400

x0 = torch.zeros((nb,1,1))
#x0[0,0,0] = 10
x_target = 10

horizon = 200
num_epochs = 5000

tau_0 = 1
tau = 0.05
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

test_data[0,70:,:] = 0.2
test_data[2,70:,:] = 0.9
test_data[1,70:,:] = 0.6
dist = [0.3,0.9,1.8]
optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)


x_log_base, u_log_base = sys.rollout(controller, tau = tau,d=test_data, neural = False)
u_out_base = sys.u_out.clone()
u_bin_base = sys.hs.clone()


best_loss = float('inf')
best_params = None

for epoch in range(num_epochs):
    
    optimizer.zero_grad()

    #tau = tau_0 * 1/np.log(0.7*epoch+2)

    # Simulate the system for 10 steps
    x_log, u_log = sys.rollout(controller, tau = tau,d=d)

    # Compute loss (mean squared error)
    target = torch.full_like(x_log,x_target)
    loss = torch.mean((x_log - target) ** 2 )
                      #+ 10*(u_log*(1-u_log)) )
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


x_log, u_log = sys.rollout(controller, tau=tau,d = d)



x_log_hard, u_log_hard = sys.rollout(controller,tau=tau,d = d, hard=True)
loss = torch.mean((x_log_hard - target) ** 2 )
print(f'Final Loss: {loss}')

x_log_test, u_log_test = sys.rollout(controller, tau=tau, d=test_data, hard=True)
u_out_test = sys.u_out.clone()
loss_test = torch.mean((x_log_test - x_target) ** 2 )
u_bin_test = sys.bs.clone()
ubs_test = sys.ubs.clone()
hs_test = sys.hs.clone()

loss_base = torch.mean((x_log_base[0] - x_target) ** 2 )
loss_test0 = torch.mean((x_log_test[0] - x_target) ** 2 )

print(f'Final test Loss: {loss_test0}')
print(f'Final base Loss: {loss_base}')

loss_base = torch.mean((x_log_base[1] - x_target) ** 2 )
loss_test0 = torch.mean((x_log_test[1] - x_target) ** 2 )

print(f'Final test Loss: {loss_test0}')
print(f'Final base Loss: {loss_base}')

loss_base = torch.mean((x_log_base[2] - x_target) ** 2 )
loss_test0 = torch.mean((x_log_test[2] - x_target) ** 2 )

print(f'Final test Loss: {loss_test0}')
print(f'Final base Loss: {loss_base}')

fig, ax = plt.subplots(3, 2, figsize=(10, 8))


row_titles = ['d=0.2', 'd=0.6', 'd=0.9']
for i, title in enumerate(row_titles):
    fig.text(0.05, 0.76 - i * 0.28, title, va='center', rotation='vertical', fontsize=12)


ax[0,1].plot(u_out_base[0,:,0].detach().numpy(), label='Bangbang')
ax[0,1].plot(u_out_test[0,:,0].detach().numpy(), label='NN')
ax[0,1].axhline(y=0.2, color='black', linestyle='--', label=r'$u_{min}$')
ax[0,1].legend()

ax[0,0].plot(x_log_base[0,:,0].detach().numpy(), label='Bangbang')
ax[0,0].plot(x_log_test[0,:,0].detach().numpy(), label='NN')
ax[0,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[0,0].legend()

ax[1,1].plot(u_out_base[1,:,0].detach().numpy(), label='Bangbang')
ax[1,1].plot(u_out_test[1,:,0].detach().numpy(), label='NN')
ax[1,1].axhline(y=0.2, color='black', linestyle='--', label=r'$u_{min}$')
ax[1,1].legend()

ax[1,0].plot(x_log_base[1,:,0].detach().numpy(), label='Bangbang')
ax[1,0].plot(x_log_test[1,:,0].detach().numpy(), label='NN')
ax[1,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[1,0].legend()

ax[2,1].plot(u_out_base[2,:,0].detach().numpy(), label='Bangbang')
ax[2,1].plot(u_out_test[2,:,0].detach().numpy(), label='NN')
ax[2,1].axhline(y=0.2, color='black', linestyle='--', label=r'$u_{min}$')
ax[2,1].legend()

ax[2,0].plot(x_log_base[2,:,0].detach().numpy(), label='Bangbang')
ax[2,0].plot(x_log_test[2,:,0].detach().numpy(), label='NN')
ax[2,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[2,0].legend()



fig, ax = plt.subplots(2, 1, figsize=(10, 4))

ax[0].plot(u_bin_base[0,:,:].detach().numpy(), label='Bangbang')
ax[0].plot(u_bin_test[0,:,:].detach().numpy(), label='NN')
ax[0].set_title('On off behavior - Base vs NN')
ax[0].legend()

ax[1].plot(hs_test[0,:,:].detach().numpy(), label='Bangbang')
ax[1].plot(ubs_test[0,:,:].detach().numpy(), label='NN Output')
ax[1].plot(u_bin_test[0,:,:].detach().numpy(), label='Used binary')
ax[1].set_title('On off behavior - NN vs Actual')
ax[1].legend()



fig, ax = plt.subplots(3, 2)

ax[0,0].plot(x_log[0,:,:].detach().numpy(), label='State x')
ax[0,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[0,0].set_title('State x')
ax[0,0].set_ylim(0, 15)
ax[0,0].legend()

ax[0,1].plot(u_log[0,:,:].detach().numpy(), label='Control u')
ax[0,1].set_title('Control u')
ax[0,1].legend()

ax[1,0].plot(x_log_hard[0,:,:].detach().numpy(), label='State x')
ax[1,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[1,0].set_ylim(0, 15)
ax[1,0].set_title('State x')
ax[1,0].legend()

ax[1,1].plot(u_log_hard[0,:,:].detach().numpy(), label='Control u')
ax[1,1].set_title('Control u')
ax[1,1].legend()




for i in range(x_log_test.shape[0]): 
    ax[2,0].plot(x_log_test[i,:,:].detach().numpy(), label='d = ' + str(dist[i]))
    ax[2,1].plot(u_log_test[i,:,:].detach().numpy(), label='d = ' + str(dist[i]))
    ax[2,0].plot(x_log_base[i,:,0].detach().numpy(), label='Bangbang')


ax[2,0].axhline(y=x_target, color='r', linestyle='--', label='Target x')
ax[2,0].set_ylim(0, 15)
ax[2,0].set_title('State x')


ax[2,1].set_title('Control u')
ax[2,1].legend()

plt.tight_layout()
plt.show()