import torch
from torch.utils.data import TensorDataset, DataLoader
from generate_data import generator_linear
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

np.random.seed(0)
learning_rate = 1e-2
D_in,H,D_out=2,10,1
epochs = 50

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

x, y = generator_linear(n = 500)
x = x.astype('float32')
y = y.astype('float32')

w1 = np.random.rand(D_in,H)
w2 = np.random.rand(H,D_out)

print(w1)
print(w2)

# print(x.dtype)
# print(y.dtype)

for epoch in range(epochs):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    loss = np.square(y_pred - y).sum()

    print("y_pred:",y_pred.shape)
    print("y:",y.shape)

    grad_y_pred = 2.0*(y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

    print("epoch:",epoch, "loss:",loss)

# plt.scatter(x[:,0],x[:,1],c = y,s=100,lw=0,cmap='coolwarm')
# plt.show()

# a = np.array([[1,2,3]])
# b = np.array([[1],[2],[3]])

# print(a)
# print(b)
#矩陣乘法
# print(a.dot(b))

# class TwoLayerNet(torch.nn.Module):
#     def __init__(self,D_in,H,D_out):
#         super(TwoLayerNet,self).__init__()
#         self.linear_1 = torch.nn.Linear(D_in, H)
#         self.linear_2 = torch.nn.Linear(H, D_out)
        
#     def forward(self, x):
#         h = self.linear_1(x)
#         h_relu = F.relu(h)
#         y_pred = self.linear_2(h_relu)
#         return y_pred

# model = TwoLayerNet(D_in = 2, H = 10, D_out = 1)    
# optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# for epoch in range(epochs):
#     y_pred = model(x)
#     loss = F.mse_loss(y_pred, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch%10==0:

#         correct= (y_pred.ge(0.5) == y).sum().item()
#         n = y.shape[0]

#         print("epochs:",epoch,"loss:",loss.item(),"Accuracy:",(correct / n))


        