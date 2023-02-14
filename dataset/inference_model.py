import torch
from torch import nn
from torchvision.models import VGG16_Weights, ResNet18_Weights, MobileNet_V3_Large_Weights
import pytorch_lightning as pl
import numpy as np

class BaseModel(pl.LightningModule):
    def __init__(self, input_dim, output_channel):
        ## output_channel: key: output_name value: output_dim
        super().__init__()
        self.net = torch.hub.load('pytorch/vision:v0.14.1', 'mobilenet_v3_large', pretrained=MobileNet_V3_Large_Weights.DEFAULT)
        self.net.classifier[3] = nn.Identity()
        w0 = self.net.features[0][0].weight.data.clone()
        self.net.features[0][0] = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.net.features[0][0].weight.data[:,:3,:,:] = w0

        self.fc1 = nn.Linear(1280 + input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
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
        print(mask.shape, image.shape)
        x = self.net(torch.cat((image, mask), dim = 1))
        x = torch.cat([x, input_vector], dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        outs = []
        for i, (output_name, output_dim) in enumerate(self.output_channel.items()):
            out =  self.output_layers[i](x)
            outs.append(out)
        return outs
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        return optimizer

    def get_loss(self, image, mask, input_vector, y):
        print('--get loss--')
        y_preds = self(image, mask, input_vector)
        
        losses = 0
        for i, (output_name, output_dim) in enumerate(self.output_channel.items()):
            if output_name == 'informativeness':
                losses += self.reg_loss(torch.round(y_preds[i]), y[:, i])
            else:
                losses += self.entropy_loss(y_preds[i], y[:,i])
        return losses

    def training_step(self, batch, batch_idx):
        image, mask, input_vector, y = batch
        loss = self.get_loss(image, mask, input_vector, y)
        return loss

    '''def validation_step(self, data, batch_idx):
        img = data['image+mask']
        y = data['label']
        loss = self.get_loss(img, y)
        # self.visualize(data, quat, state)

        return loss'''

