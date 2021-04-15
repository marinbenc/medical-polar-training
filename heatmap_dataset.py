import sys
import os.path as p

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

import albumentations.augmentations.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

import helpers as h
import polar_transformations

# based on https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/data/MPII/dp.py
class HeatmapDataset(Dataset):
  def __init__(self, dataset_name, directory, transform=None):
    self.directory = p.join('datasets', dataset_name, 'heatmap', directory)
    self.file_names = h.listdir(p.join(self.directory, 'label'))
    self.file_names.sort()

    if transform is not None:
      self.transform = A.Compose([
        transform,
        ToTensorV2()
      ])
    else:
      self.transform = ToTensorV2()

  def __len__(self):
    #return 16 # overfit single batch
    return len(self.file_names)

  def __getitem__(self, idx):
    file_name = self.file_names[idx]
    label_file = p.join(self.directory, 'label', file_name)
    input_file = p.join(self.directory, 'input', file_name)

    input = np.load(input_file)
    label = np.load(label_file)

    transformed = self.transform(image=input, mask=label)
    input, label = transformed['image'], transformed['mask']
    label = label.unsqueeze(0)

    return input, label

