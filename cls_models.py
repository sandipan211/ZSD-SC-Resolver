import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as LA

class ClsModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ClsModel, self).__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.lsm = nn.LogSoftmax(dim=1)
    def forward(self, feats=None, classifier_only=False):
        x = self.fc1(feats)
        x = self.lsm(x)
        return x
    
class ClsModelTrain(nn.Module):
    def __init__(self, num_classes=4):
        super(ClsModelTrain, self).__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    def forward(self, feats=None, classifier_only=False):
        x = self.fc1(feats)
        return x

class ClsUnseen(torch.nn.Module):
    def __init__(self, att):
        super(ClsUnseen, self).__init__()
        self.W = att.type(torch.float).cuda()
        self.fc1 = nn.Linear(in_features=1024, out_features=300, bias=True)
        self.lsm = nn.LogSoftmax(dim=1)

        print(f"__init__ {self.W.shape}")

    def forward(self, feats=None, classifier_only=False):
        f = self.fc1(feats)
        x = f.mm(self.W.transpose(1,0))
        x = self.lsm(x)

        return x

class ClsUnseenTrain(torch.nn.Module):
    def __init__(self, att):
        super(ClsUnseenTrain, self).__init__()
        self.W = att.type(torch.float).cuda()
        self.fc1 = nn.Linear(in_features=1024, out_features=300, bias=True)

        print(f"__init__ {self.W.shape}")

    def forward(self, feats=None, classifier_only=False):
        f = self.fc1(feats)
        x = f.mm(self.W.transpose(1,0))

        return x

    

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Regressor(torch.nn.Module):
    def __init__(self,in_sz=1024,out_sz=300):
        super(Regressor, self).__init__()
        
        self.fc1 = nn.Linear(in_features=in_sz, out_features=out_sz, bias=True)
        #self.fc2 = nn.Linear(in_features=662, out_features=300, bias=True)
        self.apply(weights_init)
        # self. m = nn.LeakyReLU(0.1)
        #self.m = nn.ReLU()
    def forward(self, feats=None):
        f = self.fc1(feats)
        #f=self.m(f)
        #x = self.fc2(f)
       
        return f
