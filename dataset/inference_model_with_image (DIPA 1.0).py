import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inference_dataset import ImageMaskDataset
from inference_model import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl

if __name__ == '__main__':
    bigfives = ["extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness"]
    basic_info = [ "age", "gender", "platform"]
    category = ['category']
    privacy_metrics = ['informationType', 'informativeness', 'sharing']

    mega_table = pd.read_csv('./mega_table.csv')

    encoder = LabelEncoder()
    mega_table['category'] = encoder.fit_transform(mega_table['category'])
    mega_table['gender'] = encoder.fit_transform(mega_table['gender'])
    mega_table['platform'] = encoder.fit_transform(mega_table['platform'])
    mega_table['id'] = encoder.fit_transform(mega_table['id'])

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

    image_size = (300, 300)
    label_folder = './new annotations/annotations/'
    image_folder = './new annotations/images/'
    dataset = ImageMaskDataset(mega_table, image_folder, label_folder, input_channel, output_name, image_size)

    train_loader = DataLoader(dataset, batch_size=32)

    trainer = pl.Trainer()
    trainer.fit(model, train_loader)