import itertools
from operator import itemgetter
import numpy as np
import torch
from torch._C import dtype
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import random
import time
import torchsummary
import math
from torch.tensor import Tensor
from tqdm import tqdm
from torchsummary import summary
from numba import jit
import os 
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import torch.nn.functional as F

def data_process(vocabulary, input_data, target_data, condition_data, encoder_condition_data, max_len,shuffle):
    
    for i in range(len(input_data)):

        input_data[i] = list(input_data[i])
        input_process = list(itemgetter(*list(input_data[i]))(vocabulary))

        # input_process = [0]+input_process
        input_process.insert(0,vocabulary["SOS"])
        input_process.append(vocabulary["EOS"])
        for _ in range(max_len-len(input_process)):
            input_process.insert(0,vocabulary["SOS"])
        input_process = np.array(input_process)
        input_process = input_process.astype("int64")

        input_data[i] = input_process
    
        target_data[i] = list(target_data[i])
        target_process = list(itemgetter(*list(target_data[i]))(vocabulary))

        # target_process = [0]+target_process
        target_process.insert(0,vocabulary["SOS"])
        target_process.append(vocabulary["EOS"])
        for _ in range(max_len-len(target_process)):
            target_process.insert(0,vocabulary["SOS"])
        target_process = np.expand_dims(np.array(target_process),axis=-1)
        target_process = target_process.astype("int64")

        target_data[i] = target_process

    condition_data = list(itemgetter(*list(condition_data))(vocabulary))
    encoder_condition_data = list(itemgetter(*list(encoder_condition_data))(vocabulary))

    input_data = np.array(input_data, dtype=np.int64)
    target_data = np.array(target_data, dtype=np.int64)
    condition_data = np.array(condition_data, dtype=np.int64)
    encoder_condition_data = np.array(encoder_condition_data, dtype=np.int64)

    condition_data = condition_data.reshape(condition_data.shape[0],1)
    encoder_condition_data = encoder_condition_data.reshape(encoder_condition_data.shape[0],1)

    if shuffle == True:
        np.random.seed(99)
        np.random.shuffle(input_data)
        np.random.seed(99)
        np.random.shuffle(target_data)
        np.random.seed(99)
        np.random.shuffle(condition_data)
        np.random.seed(99)
        np.random.shuffle(encoder_condition_data)

    input_data,target_data,condition_data,encoder_condition_data = torch.from_numpy(input_data),torch.from_numpy(target_data),torch.from_numpy(condition_data),torch.from_numpy(encoder_condition_data)
    input_data,target_data = Variable(input_data),Variable(target_data)

    print(input_data.shape,target_data.shape,condition_data.shape,encoder_condition_data.shape)

    return input_data, target_data, condition_data, encoder_condition_data

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size

        self.embedding = nn.Embedding(input_size,hidden_size)
        self.cond_embedding = nn.Embedding(input_size,condition_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size,input_size)

    def forward(self, input, condition, encoder_hidden, encoder_ceil, first_hidden , to_decoder):

        if  to_decoder == False:

            if first_hidden == True:
                encoder_data = self.embedding(input).view(1, 1, self.hidden_size)
                condition = self.cond_embedding(condition).view(-1,1,self.condition_size)

                encoder_hidden = torch.cat((encoder_hidden,condition),-1)
                encoder_ceil = torch.cat((encoder_ceil,condition),-1)
                # encoder_data = torch.cat((output,condition),-1)
                output, hidden = self.lstm(encoder_data,(encoder_hidden,encoder_ceil))
            else:
                encoder_data = self.embedding(input).view(1, 1, self.hidden_size)
                output, hidden = self.lstm(encoder_data,(encoder_hidden,encoder_ceil))
            
            return output, hidden

        elif to_decoder == True:
    
            hidden_mu = self.linear(encoder_hidden)
            hidden_logvar = self.linear(encoder_hidden)

            ceil_mu = self.linear(encoder_ceil)
            ceil_logvar = self.linear(encoder_ceil)

            hidden = (hidden_mu, hidden_logvar, ceil_mu, ceil_logvar)

            return hidden      

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size-self.condition_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, condition_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.cond_embedding = nn.Embedding(output_size,condition_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out1 = nn.Linear(output_size+condition_size,hidden_size)
        self.out2 = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, output, condition, hidden, ceil, first_hidden, middle_hidden):

        if first_hidden == True:
            
            output = self.embedding(input).view(1, 1, self.hidden_size)
            decoder_data = F.relu(output)
            condition = self.cond_embedding(condition).view(-1,1,self.condition_size)

            hidden = torch.cat((hidden,condition),-1)
            ceil = torch.cat((ceil,condition),-1)

            hidden = self.out1(hidden)
            ceil = self.out1(ceil)

            output, decoder_hidden = self.lstm(decoder_data,(hidden,ceil))
            output = self.softmax(self.out2(output[0]))

            return output, decoder_hidden

        elif middle_hidden == True:

            output = self.embedding(input).view(-1, 1, self.hidden_size)
            decoder_data = F.relu(output)
            output, decoder_hidden = self.lstm(decoder_data,(hidden,ceil))
            output = self.softmax(self.out2(output[0]))

            return output, decoder_hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def to_var(x):
  if torch.cuda.is_available():
      x = x.cuda()
  return Variable(x)

def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    esp = to_var(torch.randn(mu.size()))
    latent = mu + std * esp
    return latent

def tense_transfer_testing(encoder, decoder):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_data_test=['abandon','abet','begin','expend','sent','split','flared','functioning','functioning','healing']
        condition_data_test=['pp','pg','tp','tp','tp','pg','sp','sp','pp','tp']
        encoder_condition_data_test=['sp','sp','sp','pp','sp','pp','sp','pg','pg','pg']
        target_data_test=['abandoned','abetting','begins','expends','sends','splitting','flare','function','functioned','heals']

        input_data_test, target_data_test, condition_data_test, encoder_condition_data_test = data_process(vocabulary, input_data_test, target_data_test, condition_data_test, encoder_condition_data_test, max_len, False)

        input_data_test = input_data_test.to(device)
        target_data_test = target_data_test.to(device)
        condition_data_test = condition_data_test.to(device)
        encoder_condition_data_test = encoder_condition_data_test.to(device)

        result = []

        for i in range(len(input_data_test)):

            encoder_hidden1 = to_var(encoder.initHidden())

            input_tensor_test = input_data_test[i].tolist()
            target_tensor_test = target_data_test[i].tolist()

            for _ in range(input_tensor_test.count(0)-1):
                del input_tensor_test[0]

            for _ in range(target_tensor_test.count([0])-1):
                del target_tensor_test[0]

            input_length = len(input_tensor_test)
            target_length = len(target_tensor_test)
            
            input_tensor_test = torch.Tensor(input_tensor_test).long()
            input_tensor_test = to_var(input_tensor_test)

            target_tensor_test = torch.Tensor(target_tensor_test).long()
            target_tensor_test = to_var(target_tensor_test)

            condition_tensor_test = condition_data_test[i]
            # target_tensor_test = target_data_test[i]
            encoder_condition_tensor_test = encoder_condition_data_test[i]

            for char_index in range(len(input_tensor_test)):

                if char_index == 0:
                    output, encoder_hidden2 = encoder(input_tensor_test[char_index].expand(1), encoder_condition_tensor_test, encoder_hidden1, encoder_hidden1, True , False)
                else:
                    output, encoder_hidden2 = encoder(input_tensor_test[char_index].expand(1), encoder_condition_tensor_test, encoder_hidden2[0], encoder_hidden2[1], False , False)
            
            hidden_mu, hidden_logvar, ceil_mu, ceil_logvar = encoder(input_tensor_test[char_index].expand(1), encoder_condition_tensor_test, encoder_hidden2[0], encoder_hidden2[1], False ,True)
            latent_hidden = reparameterize(hidden_mu, hidden_logvar)
            latent_ceil = reparameterize(ceil_mu, ceil_logvar)

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)
            output_char=None

            predict_word = ""
            for di in range(target_length):

                if di == 0:
                    output, decoder_hidden = decoder(decoder_input, decoder_outputs, condition_tensor_test, latent_hidden, latent_ceil, True, False)
                    output_char = new_vocabulary[torch.max(output,1).indices.item()]
                    # loss += criterion(output,target_tensor[di])
                    if output_char!="SOS" and output_char!="EOS":
                        predict_word+=output_char
                    # loss += criterion(decoder_output[i].view(-1,vocab_size), target_tensor[cnt])

                else:
                    output, decoder_hidden = decoder(torch.tensor([[torch.max(output,1).indices]], device=device), decoder_outputs, condition_tensor_test, decoder_hidden[0], decoder_hidden[1], False, True)
                    output_char = new_vocabulary[torch.max(output,1).indices.item()]
                    # loss += criterion(output,target_tensor[di])
                    if output_char!="SOS" and output_char!="EOS":
                        predict_word+=output_char
                
                if output_char == "EOS":
                    break
            
            input_word = ""
            for j in range(len(input_data_test[i])):
                input_word_char = input_data_test[i][j].item()
                if input_word_char!=0 and input_word_char!=1:
                    input_word+=new_vocabulary[input_word_char]

            target_word = ""
            target_tensor_eva = target_data_test[i].reshape(target_data_test[i].shape[0],)
            for j in range(len(target_tensor_eva)):
                target_word_char = target_tensor_eva[j].item()
                if target_word_char!=0 and target_word_char!=1:
                    target_word+=new_vocabulary[target_word_char]
            
            print("Input: {:12}".format(input_word),"  Target: {:12}".format(target_word),"  Prediction: {:12}".format(predict_word))
            result.append([predict_word,target_word])

        score= []
        for i in range(len(result)):
            score.append(compute_bleu(result[i][0],result[i][1]))
        # print("Testing BLEU-4 score: ",round(sum(score)/len(score),4))

        return round(sum(score)/len(score),4)

def tense_generator_testing(encoder,decoder):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():

        data = []
        words = []

        path = "./train.txt"

        f = open(path, 'r')
        lines = f.readlines()
        for index in range(len(lines)):
            lines[index] = lines[index].strip()
            data.append(lines[index].split(" "))
            lines[index] = list(lines[index].split(" ")[0])
            lines[index] = list(itemgetter(*list(lines[index]))(vocabulary))
            lines[index].insert(0,vocabulary['SOS'])
            lines[index].append(vocabulary['EOS'])
            for j in range(max_len-len(lines[index])):
                lines[index].insert(0,vocabulary['SOS'])
        f.close()

        encoder_condition_data = ['sp','tp','pg','pp']*len(lines)
        data = data[:300]

        # print(encoder_condition_data)

        np.random.seed(99)
        np.random.shuffle(lines)
        np.random.seed(99)
        np.random.shuffle(data)
        np.random.seed(99)
        np.random.shuffle(encoder_condition_data)

        input_data_generator = lines[:100]
        data_generator = data[:100]
        encoder_condition_data_generator = encoder_condition_data[:100]

        input_data_generator = np.array(input_data_generator)

        encoder_condition_data_generator = list(itemgetter(*list(encoder_condition_data_generator))(vocabulary))
        encoder_condition_data_generator = np.array(encoder_condition_data_generator)

        input_data_generator,encoder_condition_data_generator = torch.from_numpy(input_data_generator),torch.from_numpy(encoder_condition_data_generator)
        input_data_generator,encoder_condition_data_generator = Variable(input_data_generator),Variable(encoder_condition_data_generator)
        input_data_generator,encoder_condition_data_generator = input_data_generator.to(device),encoder_condition_data_generator.to(device)
        cnt = 0

        for i in range(len(input_data_generator)):

            encoder_hidden1 = to_var(encoder.initHidden())

            # input_length = input_data_generator[i].size()

            input_tensor_generator = input_data_generator[i].tolist()

            for _ in range(input_tensor_generator.count(0)-1):
                del input_tensor_generator[0]
            
            input_tensor_generator = torch.Tensor(input_tensor_generator).long()
            input_tensor_generator = to_var(input_tensor_generator)

            encoder_condition_tensor_generator = encoder_condition_data_generator[i].long()

            for char_index in range(len(input_tensor_generator)):

                if char_index == 0:
                    output, encoder_hidden2 = encoder(input_tensor_generator[char_index].expand(1), encoder_condition_tensor_generator, encoder_hidden1, encoder_hidden1, True , False)
                else:
                    output, encoder_hidden2 = encoder(input_tensor_generator[char_index].expand(1), encoder_condition_tensor_generator, encoder_hidden2[0], encoder_hidden2[1], False , False)

            hidden_mu, hidden_logvar, ceil_mu, ceil_logvar = encoder(input_tensor_generator[char_index].expand(1), encoder_condition_tensor_generator, encoder_hidden2[0], encoder_hidden2[1], False ,True)
            latent_hidden = reparameterize(hidden_mu, hidden_logvar)
            latent_ceil = reparameterize(ceil_mu, ceil_logvar)

            # latent_hidden = to_var(torch.randn(1,1,vocab_size))
            # latent_ceil = to_var(torch.randn(1,1,vocab_size))

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)
            output_char=None

            condition_tensor_generator_data = [vocabulary['sp'],vocabulary['tp'],vocabulary['pg'],vocabulary['pp']]

            predict_list = []

            for condition in condition_tensor_generator_data:
                decoder_condition_tensor_generator = torch.Tensor([[condition]]).long().to(device)

                predict_word = ""
                for di in range(max_len):

                    if di == 0:
                        output, decoder_hidden = decoder(decoder_input, decoder_outputs, decoder_condition_tensor_generator, latent_hidden, latent_ceil, True, False)
                        output_char = new_vocabulary[torch.max(output,1).indices.item()]
                        # loss += criterion(output,target_tensor[di])
                        if output_char!="SOS" and output_char!="EOS":
                            predict_word+=output_char
                        # loss += criterion(decoder_output[i].view(-1,vocab_size), target_tensor[cnt])

                    else:
                        output, decoder_hidden = decoder(torch.tensor([[torch.max(output,1).indices]], device=device), decoder_outputs, decoder_condition_tensor_generator, decoder_hidden[0], decoder_hidden[1], False, True)
                        output_char = new_vocabulary[torch.max(output,1).indices.item()]
                        # loss += criterion(output,target_tensor[di])
                        if output_char!="SOS" and output_char!="EOS":
                            predict_word+=output_char
                        
                    if output_char=="EOS":
                        break

                predict_list.append(predict_word)
            # print("Ground_truth_list:",data[cnt])
            print(predict_list)
            # print("predict_list:",data[cnt])
            cnt+=1
            words.append(predict_list)
        gaussian_score = Gaussian_score(words)
        return gaussian_score

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
      weights = (0.33,0.33,0.33)
    else:
      weights = (0.25,0.25,0.25,0.25)
    # weights = tuple([round(1/len(reference),4)]*len(reference))

    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def Gaussian_score(words):
    words_list = []
    score = 0
    train_path = './train.txt'#should be your directory of train.txt
    with open(train_path,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


if __name__ == "__main__":

    encoderfilepath=os.path.abspath(os.path.dirname(__file__))+"\\checkpoint\\encoder.pt"
    decoderfilepath=os.path.abspath(os.path.dirname(__file__))+"\\checkpoint\\decoder.pt"

    vocabulary1={}
    vocabulary1["SOS"]=0
    vocabulary1["EOS"]=1
    vocabulary1["sp"]=28
    vocabulary1["tp"]=29
    vocabulary1["pg"]=30
    vocabulary1["pp"]=31

    vocabulary2 = {chr(number):index+2 for (index,number) in enumerate(range(97,123))}

    # vocabulary = dict(vocabulary1.items() | vocabulary2.items())
    vocabulary = {**vocabulary1, **vocabulary2}
    new_vocabulary = {v : k for k, v in vocabulary.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0
    EOS_token = 1
    #----------Hyper Parameters----------#
    hidden_size = 256
    condition_size = 8
    max_len = 17
    #The number of vocabulary
    vocab_size = 32
            
    encoder = EncoderRNN(vocab_size, hidden_size, condition_size).to(device)
    encoder.load_state_dict(torch.load(encoderfilepath))

    decoder = DecoderRNN(hidden_size, vocab_size, condition_size).to(device)
    decoder.load_state_dict(torch.load(decoderfilepath))

    bleu4_score = tense_transfer_testing(encoder,decoder)

    gaussian_score = tense_generator_testing(encoder,decoder)

    print("Testing BLEU-4 score: ",round(bleu4_score ,4))
    print("Testing Gaussian score: ",gaussian_score)

