from torch.autograd import Variable
from dataloader import read_bci_data
import pandas as pd
import torch
import numpy as np
import os

def testing(x_test,y_test,device,model):

    model.eval()
    with torch.no_grad():
        model.to(device)
        n = x_test.shape[0]

        x_test = x_test.astype("float32")
        y_test = y_test.astype("float32").reshape(y_test.shape[0],)

        x_test, y_test = Variable(torch.from_numpy(x_test)),Variable(torch.from_numpy(y_test))
        x_test,y_test = x_test.to(device),y_test.to(device)
        y_pred_test = model(x_test)
        correct = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
        print("testing accuracy:",correct/n)
        return correct/n

class EGGNet(torch.nn.Module):
    def __init__(self):
        super(EGGNet, self).__init__()
        self.firstconv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(1,51), padding=(0,25), bias=False),
            torch.nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.depthwiseConv = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            torch.nn.BatchNorm2d(32),
            # torch.nn.ELU(alpha=1.0),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            torch.nn.Dropout(p=0.35),
        )
        self.separableConv = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            torch.nn.BatchNorm2d(32),
            # torch.nn.ELU(alpha=1.0),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            torch.nn.Dropout(p=0.45),
        )
        self.classify = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=736, out_features=2, bias=True),
        )
    
    def forward(self, input_tensor):
        x = self.firstconv(input_tensor)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x

if __name__ == "__main__":
    filepath=os.path.abspath(os.path.dirname(__file__))+"\\checkpoint\\model_best_acc_9027_relu.pt"
    device = torch.device("cuda:0")

    # model = EGGNet()
    # model.load_state_dict(torch.load(filepath))

    train_data, train_label, test_data, test_label = read_bci_data()
    test_data/=6.6221
    print(np.std(train_data),np.min(train_data))
    print(np.max(test_data),np.min(test_data))

    # testing_accuracy = testing(test_data,test_label,device,model)