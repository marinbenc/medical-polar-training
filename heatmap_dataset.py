import sys
import os.path as p

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import polar_transformations

# based on https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/data/MPII/dp.py
class HeatmapDataset(Dataset):
  def __init__(self, wrapped_dataset, manual_centers=None):
    self.manual_centers = manual_centers
    self.wrapped_dataset = wrapped_dataset

  @staticmethod
  def _generate_heatmap(output_res, center):
    sigma = max(output_res) / 32
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3*sigma + 1, 3*sigma + 1

    # gaussian
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    hms = np.zeros(output_res, dtype=np.float32)
    
    x, y = int(center[0]), int(center[1])

    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

    c, d = max(0, -ul[0]), min(br[0], output_res[1]) - ul[0]
    a, b = max(0, -ul[1]), min(br[1], output_res[1]) - ul[1]

    cc, dd = max(0, ul[0]), min(br[0], output_res[1])
    aa, bb = max(0, ul[1]), min(br[1], output_res[1])

    hms[aa:bb,cc:dd] = np.maximum(hms[aa:bb,cc:dd], g[a:b,c:d])

    return hms

  def __len__(self):
    return self.wrapped_dataset.__len__()

  def __getitem__(self, idx):
    (input, label) = self.wrapped_dataset.__getitem__(idx)

    input = torchvision.transforms.functional.resize(input, (256, 256))
    label = torchvision.transforms.functional.resize(label, (256, 256))

    if self.manual_centers is not None:
      center = self.manual_centers[idx]
    else:
      label = label.detach().numpy().squeeze()
      center = polar_transformations.centroid(label)

    label = HeatmapDataset._generate_heatmap(label.shape[-2:], center)
    label = torch.from_numpy(label).unsqueeze(0)
    label = torchvision.transforms.functional.resize(label, (64, 64))
    return input, label

