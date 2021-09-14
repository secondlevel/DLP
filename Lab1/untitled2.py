# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:01:35 2021

@author: haoyuan
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from generate_data import generator_linear
from torch.optim.lr_scheduler import StepLR
import numpy as np

torch.manual_seed(1)    # reproducible
data_number = 500.
epochs = 100
torch.cuda.set_device(0)

x, y = generator_linear(n = int(data_number))
x = torch.from_numpy(x).float()
y = np.squeeze(y, 1).astype('int64')
y = torch.from_numpy(y)

print(x)
print(y)
# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)
# print(x)
# print(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='coolwarm')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        h = self.hidden(x)
        x = F.relu(h)      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
# optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted\


# plt.ion()   # something about plotting

# for t in range(epochs):
#     out = net(x)                 # input x and predict based on x
#     loss = loss_func(out, y)# must be (1. nn output, 2. target), the target label is NOT one-hotted

#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()        # apply gradients
    
#     if t % 10 == 0:
#         # plot and show learning process
#         plt.cla()
#         plt.subplot(1,2,1)
#         plt.title("Ground Truth",fontsize=18)
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='coolwarm')

#         _, prediction = torch.max(F.softmax(out), 1)
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = y.data.numpy()

#         if t==90:
#             print("pred_y:",pred_y)
#             print("target_y:",target_y)

#         plt.subplot(1,2,2)
#         plt.title("Predict Result",fontsize=18)
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='coolwarm')
#         accuracy = sum(pred_y == target_y)/data_number
#         print('epochs:',t,'loss:',loss.item(),'accuracy:',accuracy)
#         plt.show()
#         plt.pause(1.0)

#         print(pred_y.shape)

#         scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
#         scheduler.step()
        
# plt.ioff()


