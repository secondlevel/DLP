import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def to_var(x):
  if torch.cuda.is_available():
      x = x.cuda()
  return Variable(x)

def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    esp = torch.randn(*mu.size())
    latent = mu + std * esp
    return latent

lstm = nn.LSTM(320, 320) #hidden_size+condition_dims   -->encoder
lstm1 = nn.LSTM(96, 96)  #vocuba_size+condition_dims   -->decoder
embed = nn.Embedding(32,256) #(vocuba_size,hidden_size) -->encoder
embed1 = nn.Embedding(32,32) #(vocuba_size,vocuba_size) -->decoder
embed_cond = nn.Embedding(32,64) #(vocuba_size,condition_dims) -->encoder -->decoder

linear = nn.Linear(320,32) #(hidden_size+condition_dims,vocuba_size) -->encoder
linear1 = nn.Linear(96,256) #(vocuba_size+condition_dims,hidden_size) -->decoder

a = torch.rand((1,16,256))
b = torch.Tensor([ 0,  0,  0,  0,  0,  0,  0,  2,  5, 11, 16, 22, 19, 15,  6,  5]).long()
c = torch.Tensor([31]).long()

c = embed_cond(c).view(-1,1,64).repeat(1, 16, 1)
print("c:",c.shape)

encoder_hidden = torch.cat((a,c),-1)
print("encoder_hidden:",encoder_hidden.shape)

input_data = embed(b).view(1,16,-1)
print("input_data:",input_data.shape)

encoder_data = torch.cat((input_data,c),-1)
print("encoder_data:",encoder_data.shape)

hidden = lstm(encoder_data,(encoder_hidden,encoder_hidden))[1][0]
ceil = lstm(encoder_data,(encoder_hidden,encoder_hidden))[1][1]

hidden0 = linear(hidden)
hidden1 = linear(hidden)
print("hidden0:",hidden0.shape)
print("hidden1:",hidden1.shape)

hidden = reparameterize(hidden0,hidden1)

print("hidden:",hidden.shape)

decoder_hidden = torch.cat((hidden,c),-1)
print("decoder_hidden:",decoder_hidden.shape)

input_data = embed1(b).view(1,16,-1)
print("input_data:",input_data.shape)

decoder_data = torch.cat((input_data,c),-1)
print("decoder_data:",decoder_data.shape)

hidden = lstm1(decoder_data,(decoder_hidden,decoder_hidden))[1][0]
ceil = lstm1(decoder_data,(decoder_hidden,decoder_hidden))[1][1]

hidden0 = linear1(hidden)
hidden1 = linear1(hidden)

print(hidden0.shape)
print(hidden1.shape)

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 


a = frange_cycle_linear(147240, start=0.0, stop=1.0,  n_cycle=147240//10000, ratio=0.5)
b = frange_cycle_linear(147240, start=0.0, stop=1.0,  n_cycle=1, ratio=0.25)

147240

print(a)
print(b)

plt.plot(b)

plt.xticks( (0, 5*1000, 10*1000, 15*1000, 20*1000, 25*1000, 30*1000, 35*1000, 40*1000), \
           ('0','5K','10K','15K','20K','25K','30K','35K','40K'), color='k', size=14)

plt.yticks((0.0, 0.5, 1.0), ('0', '0.5','1'), color='k', size=14)

plt.show()