import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import json


class ImageMaskDataset(Dataset):
    def __init__(self, mega_table, image_folder, label_folder, input_vector, output_vector, image_size, flip_prob = 0.5):
        self.mega_table = mega_table
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.input_vector = input_vector
        self.output_vector = output_vector
        self.image_size = image_size
        self.flip_prob = flip_prob
        self.padding_color = (0, 0, 0)

    def __len__(self):
        return len(self.mega_table)

    def __getitem__(self, idx):
        
        image_path = self.mega_table['imagePath'].iloc[idx]
        image = Image.open(os.path.join(self.image_folder, image_path)).convert('RGB')
        w, h = image.size
        ratio = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = TF.resize(image, (new_h, new_w))
        image = TF.pad(image, padding=(0, 0, self.image_size[1] - new_w, self.image_size[0] - new_h), fill=self.padding_color)
        image = TF.to_tensor(image)

        trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ## generate mask
        image = trans(image)

        category = self.mega_table['originCategory'].iloc[idx]
        label_file = image_path[:-4] + '_label.json'
        labels = None
        bboxes = []
        with open(os.path.join(self.label_folder, label_file)) as f:
            labels = json.load(f)
        for key, value in labels['annotations'].items():
            if value['category'] == category:
                x, y, w, h = value['bbox']
                x = x * ratio
                y = y * ratio
                w = w * ratio
                h = h * ratio
                bboxes.append([x,y,w,h])

        mask = torch.zeros((self.image_size[0], self.image_size[1]), dtype=torch.uint8)
        for x, y, w, h in bboxes:
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask[y:y+h, x:x+w] = 255
        #input vector
        if torch.rand(1) < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        mask = mask.unsqueeze(0)
        input_vector = self.mega_table[self.input_vector].iloc[idx].values
        input_vector = torch.from_numpy(input_vector)
        #label
        label = self.mega_table[self.output_vector].iloc[idx].values
        label = torch.from_numpy(label)
        return image, mask, input_vector, label


if __name__ == '__main__':
    image_size = (512, 512)
    mega_table = pd.read_csv('./mega_table.csv')
    label_folder = './new annotations/annotations/'
    image_folder = './new annotations/images/'
    input_vector =  ['informationType', 'informativeness']
    output_vector = ['sharing']
    dataset = ImageMaskDataset(mega_table, image_folder, label_folder, input_vector, output_vector, image_size)

    # Print the length of the dataset
    print('Dataset length:', len(dataset))

    # Print some sample data from the dataset
    for i in range(5):
        sample = dataset[i]
        image, mask, input_vector, label = sample
        print('Sample', i, 'image shape:', image.shape)
        print('Sample', i, 'label shape:', mask.shape)
        print('Sample', i, 'image shape:', input_vector.shape)
        print('Sample', i, 'label shape:', label.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(TF.to_pil_image(image))
        plt.subplot(1, 2, 2)
        plt.imshow(TF.to_pil_image(mask), cmap='gray', vmin=0, vmax=255)
        plt.show()