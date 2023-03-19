import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inference_dataset import ImageMaskDataset
from inference_model import BaseModel

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from sklearn import metrics

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError

def l1_distance_loss(prediction, target):
    loss = np.abs(prediction - target)
    return np.mean(loss)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if __name__ == '__main__':
    bigfives = ["extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness"]
    basic_info = [ "age", "gender", 'datasetName', 'platform']
    category = ['category']
    privacy_metrics = ['informationType', 'informativeness', 'sharing']

    mega_table = pd.read_csv('./mega_table.csv')

    encoder = LabelEncoder()
    mega_table['category'] = encoder.fit_transform(mega_table['category'])
    mega_table['gender'] = encoder.fit_transform(mega_table['gender'])
    mega_table['platform'] = encoder.fit_transform(mega_table['platform'])
    mega_table['datasetName'] = encoder.fit_transform(mega_table['datasetName'])

    input_channel = []
    input_channel.extend(basic_info)
    input_channel.extend(category)
    input_channel.extend(bigfives)
    input_dim = len(input_channel)
    output_name = privacy_metrics
    output_channel = {}
    for output in output_name:
        output_channel[output] = len(mega_table[output].unique())
    
    model = BaseModel(input_dim= input_dim, output_channel = output_channel)

    image_size = (224, 224)
    label_folder = './new annotations/annotations/'
    image_folder = './new annotations/images/'

    num_rows = len(mega_table)
    train_size = int(0.8 * num_rows)
    test_size = num_rows - train_size

    # prolific_data = mega_table[mega_table['platform'] == 1]
    # Crowdworks_data = mega_table[mega_table['platform'] == 0]
    # Split the dataframe into two
    train_df = mega_table.sample(n=train_size, random_state=0)
    val_df = mega_table.drop(train_df.index)

    # train_df = Crowdworks_data
    # val_df = prolific_data.sample(n = int(0.2 * len(train_df)), random_state=0)

    train_dataset = ImageMaskDataset(train_df, image_folder, label_folder, input_channel, output_name, image_size, flip = True)
    val_dataset = ImageMaskDataset(val_df, image_folder, label_folder, input_channel, output_name, image_size)    

    train_loader = DataLoader(train_dataset, batch_size=96, generator=torch.Generator(device='cuda'), shuffle=True)
    val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=32)
    
    wandb_logger = WandbLogger(project="DIPA2.0-inference test", name = 'DIPA 1.0: with no privacy (resnet 50)')
    checkpoint_callback = ModelCheckpoint(dirpath='./models/DIPA 1.0: with no privacy (resnet 50)/', save_last=True, monitor='val loss')

    trainer = pl.Trainer(accelerator='gpu', devices=[0],logger=wandb_logger, 
    auto_lr_find=True, max_epochs = 100, callbacks=[checkpoint_callback])
    lr_finder = trainer.tuner.lr_find(model, train_loader)
    model.hparams.learning_rate = lr_finder.suggestion()
    print(f'lr auto: {lr_finder.suggestion()}')
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    
    # validation. 
    # I am confused about how the validation_step work on saving all valid result (rather than just a batch)
    # So I wrote this traditional one

    output_channel = {'informationType': 5, 'sharingOwner': 5}

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
        image, mask, information, informativeness, sharingOwner = vdata
        y_preds = model(image.to('cuda'), mask.to('cuda'))

        acc[0].update(y_preds[:, :5], information.to('cuda'))
        pre[0].update(y_preds[:, :5], information.type(torch.FloatTensor).to('cuda'))
        rec[0].update(y_preds[:, :5], information.type(torch.FloatTensor).to('cuda'))
        f1[0].update(y_preds[:, :5], information.type(torch.FloatTensor).to('cuda'))
        conf[0].update(y_preds[:, :5], information.type(torch.LongTensor).to('cuda'))

        distance += l1_distance_loss(informativeness.detach().cpu().numpy(), y_preds[:,5].detach().cpu().numpy())

        acc[1].update(y_preds[:, 6:11], sharingOwner.to('cuda'))
        pre[1].update(y_preds[:, 6:11], sharingOwner.type(torch.FloatTensor).to('cuda'))
        rec[1].update(y_preds[:, 6:11], sharingOwner.type(torch.FloatTensor).to('cuda'))
        f1[1].update(y_preds[:, 6:11], sharingOwner.type(torch.FloatTensor).to('cuda'))
        conf[1].update(y_preds[:, 6:11], sharingOwner.type(torch.LongTensor).to('cuda'))


    distance = distance / len(val_loader)

    pandas_data = {'Accuracy' : [i.compute().detach().cpu().numpy() for i in acc], 
                   'Precision' : [i.compute().detach().cpu().numpy() for i in pre], 
                   'Recall': [i.compute().detach().cpu().numpy() for i in rec], 
                   'f1': [i.compute().detach().cpu().numpy() for i in f1]}
    
    print(pandas_data.round(3))

    for i, (output_name, output_dim) in enumerate(output_channel.items()):
        with open('./confusion {}'.format(output_name), 'w') as w:
            w.write(str(conf[i].compute().detach().cpu().numpy()))