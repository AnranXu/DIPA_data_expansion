import torch
from torch import nn
from torchvision.models import VGG16_Weights, ResNet18_Weights, MobileNet_V3_Large_Weights
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics

class BaseModel(pl.LightningModule):
    def __init__(self, input_dim, output_channel):
        ## output_channel: key: output_name value: output_dim
        super().__init__()
        self.net = torch.hub.load('pytorch/vision:v0.14.1', 'resnet18', pretrained=ResNet18_Weights.DEFAULT)
        self.net.fc = nn.Identity()
        w0 = self.net.conv1.weight.data.clone()
        self.net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.conv1.weight.data[:,:3,:,:] = w0

        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100 + input_dim, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.output_layers = []
        self.output_channel = output_channel
        for output_name, output_dim in self.output_channel.items():
            if output_name == 'informativeness':
                self.output_layers.append(nn.Linear(32,1))
            else:
                self.output_layers.append(nn.Linear(32,output_dim))
        self.act = nn.SiLU()
        self.reg_loss = nn.L1Loss()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, image, mask, input_vector):
        # x: [bs, 4, imgsize, imgsize]
        # addition: [bs, featurelength]
        x = self.net(torch.cat((image, mask), dim = 1))
        x = self.act(self.fc1(x))
        x = torch.cat([x, input_vector], dim=1)
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        outs = []
        for i, (output_name, output_dim) in enumerate(self.output_channel.items()):
            out =  self.output_layers[i](x)
            outs.append(out)
        return outs
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        return optimizer

    def get_loss(self, image, mask, input_vector, y):
        print('--get loss--')
        y_preds = self(image, mask, input_vector)
        losses = 0
        for i, (output_name, output_dim) in enumerate(self.output_channel.items()):
            print(output_name)
            if output_name == 'informativeness':
                # map label 0~6 to 0~1
                losses += self.reg_loss(torch.round(y_preds[i]).squeeze(1), y[:, i])
            else:
                losses += self.entropy_loss(y_preds[i], y[:,i].type(torch.LongTensor))
        return losses

    def training_step(self, batch, batch_idx):
        image, mask, input_vector, y = batch
        image = image.to('cuda')
        mask = mask.to('cuda')
        input_vector = input_vector.to('cuda')
        y = y.to('cuda')
        loss = self.get_loss(image, mask, input_vector, y)
        return loss

    '''def validation_step (self, val_batch, batch_idx):
        def l1_distance_loss(prediction, target):
            loss = np.abs(prediction - target)
            return np.mean(loss)

        image, mask, input_vector, y = val_batch
        y_preds = self(image, mask, input_vector)
        acc = np.zeros(len(self.output_channel))
        pre = np.zeros(len(self.output_channel))
        rec = np.zeros(len(self.output_channel))
        f1 = np.zeros(len(self.output_channel))
        distance = 0.0
        conf = []
        for i, (output_name, output_dim) in enumerate(self.output_channel.items()):
            conf.append(np.zeros((output_dim,output_dim)))
        for j, (output_name, output_dim) in enumerate(self.output_channel.items()):
            
            if output_name == 'informativeness':
                distance += l1_distance_loss(y[:, j].detach().cpu().numpy(), y_preds[j].detach().cpu().numpy())
            else:
                _, max_indices = torch.max(y_preds[j], dim = 1)
                acc[j] += metrics.accuracy_score(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy())
                pre[j] += metrics.precision_score(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(),average='weighted')
                rec[j] += metrics.recall_score(y[:, j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(),average='weighted')
                f1[j] += metrics.f1_score(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(),average='weighted')
                conf[j] += metrics.confusion_matrix(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(), labels = np.arange(0,output_dim))
                
        pandas_data = {'Accuracy' : acc, 'Precision' : pre, 'Recall': rec, 'f1': f1}
        df = pd.DataFrame(pandas_data, index=self.output_channel.keys())
        print(df.round(3))
        if 'informativeness' in self.output_channel.keys():
            print('informativenss distance: ', distance)'''
