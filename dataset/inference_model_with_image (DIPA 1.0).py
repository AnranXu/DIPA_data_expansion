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

def l1_distance_loss(prediction, target):
    loss = np.abs(prediction - target)
    return np.mean(loss)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if __name__ == '__main__':
    bigfives = ["extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness"]
    basic_info = [ "age", "gender", 'datasetName']
    category = ['category']
    privacy_metrics = ['informationType', 'informativeness', 'sharing']

    mega_table = pd.read_csv('./mega_table.csv')

    encoder = LabelEncoder()
    mega_table['category'] = encoder.fit_transform(mega_table['category'])
    mega_table['gender'] = encoder.fit_transform(mega_table['gender'])
    mega_table['platform'] = encoder.fit_transform(mega_table['platform'])
    mega_table['id'] = encoder.fit_transform(mega_table['id'])
    mega_table['datasetName'] = encoder.fit_transform(mega_table['datasetName'])
    mega_table['informativeness'] = mega_table['informativeness']

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

    # Split the dataframe into two
    train_df = mega_table.sample(n=train_size, random_state=0)
    val_df = mega_table.drop(train_df.index)

    train_dataset = ImageMaskDataset(train_df, image_folder, label_folder, input_channel, output_name, image_size, flip = True)
    val_dataset = ImageMaskDataset(val_df, image_folder, label_folder, input_channel, output_name, image_size)    

    train_loader = DataLoader(train_dataset, batch_size=96, generator=torch.Generator(device='cuda'), shuffle=True)
    val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'), batch_size=64)
    
    wandb_logger = WandbLogger(project="DIPA-inference", name = 'mix losses without nationality (resnet 50)')
    checkpoint_callback = ModelCheckpoint(dirpath='./models/mix losses without nationality (resnet 50)/', save_last=True, monitor='val loss')

    trainer = pl.Trainer(accelerator='gpu', devices=[1],logger=wandb_logger, 
    auto_lr_find=True, max_epochs = 300, callbacks=[checkpoint_callback])
    lr_finder = trainer.tuner.lr_find(model, train_loader)
    model.hparams.learning_rate = lr_finder.suggestion()
    print(f'lr auto: {lr_finder.suggestion()}')
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    
    # validation. 
    # I am confused about how the validation_step work on saving all valid result (rather than just a batch)
    # So I wrote this traditional one
    acc = np.zeros(len(output_channel))
    pre = np.zeros(len(output_channel))
    rec = np.zeros(len(output_channel))
    f1 = np.zeros(len(output_channel))
    distance = 0.0
    conf = []
    for i, (output_name, output_dim) in enumerate(output_channel.items()):
        conf.append(np.zeros((output_dim,output_dim)))

    for i, vdata in enumerate(val_loader):
        image, mask, input_vector, y = vdata
        y_preds = model(image, mask, input_vector)
        for j, (output_name, output_dim) in enumerate(output_channel.items()):
            _, max_indices = torch.max(y_preds[j], dim = 1)
            acc[j] += metrics.accuracy_score(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy())
            pre[j] += metrics.precision_score(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(),average='weighted')
            rec[j] += metrics.recall_score(y[:, j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(),average='weighted')
            f1[j] += metrics.f1_score(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(),average='weighted')
            conf[j] += metrics.confusion_matrix(y[:,j].detach().cpu().numpy(), max_indices.detach().cpu().numpy(), labels = mega_table[output_name].unique())
            if output_name == 'informativeness':
                distance += l1_distance_loss(y[:, j].detach().cpu().numpy(), max_indices.detach().cpu().numpy())
    
    ## save result
    length = len(val_dataset)
    acc = acc / length
    pre = pre / length
    rec = rec / length
    f1 = f1 / length
    distance = distance / length

    pandas_data = {'Accuracy' : acc, 'Precision' : pre, 'Recall': rec, 'f1': f1}
    df = pd.DataFrame(pandas_data, index=output_channel.keys())
    print(df.round(3))
    if 'informativeness' in output_channel.keys():
        print('informativenss distance: ', distance)