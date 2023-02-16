
import torch
import torch.nn as nn
from torch.utils.data import Dataset
class nn_model(nn.Module):
    def __init__(self,input_dim,output_dims):
        super(nn_model,self).__init__()
        self.input_layer    = nn.Linear(input_dim,128)
        self.hidden_layer1  = nn.Linear(128,256)
        self.hidden_layer2  = nn.Linear(256,128)
        self.hidden_layer3  = nn.Linear(128,64)
        self.output_layers = []
        self.output_dims = output_dims
        for i, dim in enumerate(output_dims):
            self.output_layers.append(nn.Linear(64,dim))
        self.relu = nn.SiLU()
    
    
    def forward(self,x):
        front =  self.relu(self.input_layer(x))
        front =  self.relu(self.hidden_layer1(front))
        front =  self.relu(self.hidden_layer2(front))
        front =  self.relu(self.hidden_layer3(front))
        outs = []
        for i, dim in enumerate(self.output_dims):
            out =  self.output_layers[i](front)
            outs.append(out)
        return outs
'''class nn_model(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(nn_model,self).__init__()
        self.input_layer    = nn.Linear(input_dim,128)
        self.hidden_layer1  = nn.Linear(128,64)
        self.output_layer   = nn.Linear(64,output_dim)
        self.relu = nn.ReLU()
    
    
    def forward(self,x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.hidden_layer1(out))
        out =  self.output_layer(out)
        return out'''
class nn_dataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

