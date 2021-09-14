# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:44:06 2021

@author: haoyuan
"""

import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0")
learning_rate = 1e-2

x = torch.randn(64,1000,device=device)
y = torch.randn(64,10,device=device)

# print(x)
# print(y)

loader = DataLoader(TensorDataset(x,y),batch_size=8)

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear_1 = torch.nn.Linear(D_in, H)
        self.linear_2 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        h = self.linear_1(x)
        h_relu = torch.nn.functional.relu(h)
        y_pred = self.linear_2(h_relu)
        return y_pred
    
model = TwoLayerNet(D_in = 1000, H = 100, D_out = 10)    
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


for epochs in range(50):
    for x_batch,y_batch in loader:
        y_pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        print("epoch:",epochs,"loss:",loss.item())
        # print("y_pred:",y_pred)
        # print("y_batch:",y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
        
        

