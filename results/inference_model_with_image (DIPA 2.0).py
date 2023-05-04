import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inference_dataset import ImageMaskDataset
from inference_model import BaseModel

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import json

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def l1_distance_loss(prediction, target):
    loss = np.abs(prediction - target)
    return np.mean(loss)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if __name__ == '__main__':
    bigfives = ["extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness"]
    basic_info = [ "age", "gender", 'originalDataset', 'nationality']
    category = ['category']
    privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']

    mega_table = pd.read_csv('./annotations.csv')

    description = {'informationType': ['It tells personal information', 'It tells location of shooting',
        'It tells individual preferences/pastimes', 'It tells social circle', 
        'It tells others\' private/confidential information', 'Other things'],
        'informativeness':['Strongly disagree','Disagree','Slightly disagree','Neither',
        'Slightly agree','Agree','Strongly agree'],
        'sharingOwner': ['I won\'t share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'],
        'sharingOthers': ['I won\'t allow others to share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients']}
    
    encoder = LabelEncoder()
    mega_table['category'] = encoder.fit_transform(mega_table['category'])
    mega_table['gender'] = encoder.fit_transform(mega_table['gender'])
    mega_table['platform'] = encoder.fit_transform(mega_table['platform'])
    # mega_table['id'] = encoder.fit_transform(mega_table['id'])
    mega_table['datasetName'] = encoder.fit_transform(mega_table['datasetName'])
    mega_table['nationality'] = encoder.fit_transform(mega_table['datasetName'])

    input_channel = []
    input_channel.extend(basic_info)
    input_channel.extend(category)
    input_channel.extend(bigfives)
    input_dim = len(input_channel)
    output_name = privacy_metrics
    output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}


    model = BaseModel(input_dim= input_dim)

    image_size = (224, 224)
    label_folder = './new annotations/annotations/'
    image_folder = './new annotations/images/'

    num_rows = len(mega_table)
    train_size = int(0.8 * num_rows)
    test_size = num_rows - train_size

    # Split the dataframe into two
    train_df = mega_table.sample(n=train_size, random_state=0)
    val_df = mega_table.drop(train_df.index)

    train_dataset = ImageMaskDataset(train_df, image_folder, label_folder, input_channel, image_size, flip = True)
    val_dataset = ImageMaskDataset(val_df, image_folder, label_folder, input_channel, image_size)    

    train_loader = DataLoader(train_dataset, batch_size=96, generator=torch.Generator(device='cuda'), shuffle=True)
    val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=32)
    
    wandb_logger = WandbLogger(project="DIPA2.0-inference test", name = 'only category (resnet 50)')
    checkpoint_callback = ModelCheckpoint(dirpath='./models/only category (resnet 50)/', save_last=True, monitor='val loss')

    trainer = pl.Trainer(accelerator='gpu', devices=[0],logger=wandb_logger, 
    auto_lr_find=True, max_epochs = 100, callbacks=[checkpoint_callback])
    lr_finder = trainer.tuner.lr_find(model, train_loader)
    model.hparams.learning_rate = lr_finder.suggestion()
    print(f'lr auto: {lr_finder.suggestion()}')
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader) #, ckpt_path="./models/200 epoch (resnet 50)/epoch=13-step=700.ckpt")
    
    # validation. 
    # I am confused about how the validation_step work on saving all valid result (rather than just a batch)
    # So I wrote this traditional one
    threshold = 0.5
    average_method = 'micro'
    acc = [Accuracy(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    pre = [Precision(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    rec = [Recall(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    f1 = [F1Score(task="multilabel", num_labels=output_dim, threshold = threshold, average=average_method, ignore_index = output_dim - 1) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    conf = [ConfusionMatrix(task="multilabel", num_labels=output_dim) \
            for i, (output_name, output_dim) in enumerate(output_channel.items())]
    distance = 0.0
    
    model.to('cuda')
    for i, vdata in enumerate(val_loader):
        image, mask, information, informativeness, sharingOwner, sharingOthers = vdata
        y_preds = model(image.to('cuda'), mask.to('cuda'))
        print(y_preds[:, :6].shape, information.shape)

        acc[0].update(y_preds[:, :6], information.to('cuda'))
        pre[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        rec[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        f1[0].update(y_preds[:, :6], information.type(torch.FloatTensor).to('cuda'))
        conf[0].update(y_preds[:, :6], information.to('cuda'))

        distance += l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,6].detach().cpu().numpy())
        # acc[1](y_preds[:, 6], informativeness.type(torch.FloatTensor).to('cuda'))
        # pre[1](y_preds[:, 6], informativeness.type(torch.FloatTensor).to('cuda'))
        # rec[1](y_preds[:, 6], informativeness.type(torch.FloatTensor).to('cuda'))
        # f1[1](y_preds[:, 6], informativeness.type(torch.FloatTensor).to('cuda'))
        # conf[1](y_preds[:, 6], informativeness.type(torch.FloatTensor).to('cuda'))

        acc[1].update(y_preds[:, 7:14], sharingOwner.to('cuda'))
        pre[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        rec[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        f1[1].update(y_preds[:, 7:14], sharingOwner.type(torch.FloatTensor).to('cuda'))
        conf[1].update(y_preds[:, 7:14], sharingOwner.to('cuda'))

        acc[2].update(y_preds[:, 14:21], sharingOthers.to('cuda'))
        pre[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        rec[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        f1[2].update(y_preds[:, 14:21], sharingOthers.type(torch.FloatTensor).to('cuda'))
        conf[2].update(y_preds[:, 14:21], sharingOthers.to('cuda'))


    distance = distance / len(val_loader)

    pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in acc], 
                   'Precision' : [i.compute().detach().cpu().numpy() for i in pre], 
                   'Recall': [i.compute().detach().cpu().numpy() for i in rec], 
                   'f1': [i.compute().detach().cpu().numpy() for i in f1]}

    print(pandas_data)

    for i, (output_name, output_dim) in enumerate(output_channel.items()):
        with open('./confusion {}'.format(output_name), 'w') as w:
            w.write(str(conf[i].compute().detach().cpu().numpy()))
    #     #conf[i] = conf[i].astype('float') / conf[i].sum(axis=1)[:, np.newaxis]
    #     print(conf[i].compute().detach().cpu().numpy())
    #     plt.imshow(conf[i].compute().detach().cpu().numpy(), cmap=plt.cm.Blues)
    #     plt.xticks(np.arange(0, len(description[output_name])), description[output_name], rotation = 45, ha='right')
    #     plt.yticks(np.arange(0, len(description[output_name])), description[output_name])
    #     plt.xlabel("Predicted Label")
    #     plt.ylabel("True Label")
    #     plt.title('confusion matrix for {}'.format(output_name))
    #     plt.colorbar()
    #     plt.tight_layout()

    #     #plt.savefig('confusion matrix for {}.png'.format(output_name), dpi=1200)
    #     img_buf = io.BytesIO()
    #     plt.savefig(img_buf, format='png', dpi=1200)
    #     im = Image.open(img_buf)
    #     image = wandb.Image(im, caption='confusion matrix for {}'.format(output_name))
    #     wandb_logger.log({'confusion matrix for {}'.format(output_name): image})
    #     plt.clf()
    #     print('confusion matrix for {}'.format(output_name))
    #     print(np.round(conf[i].compute().detach().cpu().numpy(), 3))

    df = pd.DataFrame(pandas_data, index=output_channel.keys())
    print(df.round(3))
    df.to_csv('./result.csv', index =False)
    with open('./distance', 'w') as w:
        w.write(str(distance))
    
    if 'informativeness' in output_channel.keys():
        print('informativenss distance: ', distance)