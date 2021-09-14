import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


embed = nn.Embedding(32,256) #(vocuba_size,hidden_size) -->encoder
embed1 = nn.Embedding(32,32) #(vocuba_size,vocuba_size) -->decoder
embed_cond = nn.Embedding(32,8) #(vocuba_size,condition_dims) -->encoder -->decoder

a = torch.Tensor([2]).long()
c = torch.Tensor([31]).long()

lstm = nn.LSTM(320, 320) #hidden_size+condition_dims   -->encoder
lstm1 = nn.LSTM(96, 96)  #vocuba_size+condition_dims   -->decoder

encoder_condition = embed_cond(c).view(-1,1,8)

encoder_init_hidden = torch.rand((1,1,256))

encoder_hidden = torch.cat((encoder_init_hidden,encoder_condition),-1)

encoder_embed_data = embed(a).view(-1,1,256)
encoder_data = torch.cat((encoder_embed_data,encoder_condition),-1)








print(encoder_hidden.shape)


