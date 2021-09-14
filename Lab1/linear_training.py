import torch
from torch.utils.data import TensorDataset, DataLoader
from generate_data import generator_linear,generator_XOR_easy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# device = torch.device("cuda:0")
learning_rate = 1e-2
epochs = 500
torch.cuda.set_device(0)

# x, y = generator_linear(n = 500)
x, y = generator_XOR_easy(n = 11)
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# loader = DataLoader(TensorDataset(x,y),batch_size=50) 
plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c = y.data.numpy(),s=100,lw=0,cmap='coolwarm')
plt.show()

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear_1 = torch.nn.Linear(D_in, H)
        self.linear_2 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        h = self.linear_1(x)
        h_relu = F.relu(h)
        y_pred = self.linear_2(h_relu)
        return y_pred

model = TwoLayerNet(D_in = 2, H = 10, D_out = 1)    
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(epochs):
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%10==0:

        correct= (y_pred.ge(0.5) == y).sum().item()
        n = y.shape[0]

        print("epochs:",epoch,"loss:",loss.item(),"Accuracy:",(correct / n))

# y_pred_total=[]
# y_answ_total=[]

# for epoch in range(epochs):

#     for (x_batch,y_batch) in loader:
#         y_pred = model(x_batch)
#         loss = F.mse_loss(y_pred, y_batch)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         y_pred_total+=list(y_pred.ge(0.5).numpy())
#         y_answ_total+=list(y_batch.numpy())

#     if epoch%10==0:

#         correct= (np.array(y_pred_total) == np.array(y_answ_total)).sum().item()
#         n = np.array(y_answ_total).shape[0]

#         print("epochs:",epoch,"loss:",loss.item(),"Accuracy:",round((correct / n),3))
        