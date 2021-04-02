import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class BBOXDataset(Dataset):
  def __init__(self, wrapped_dataset):
    self.wrapped_dataset = wrapped_dataset

  @staticmethod
  def _get_bounding_box(mask):
    # https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/utils.py
    mask = mask.type(torch.uint8)
    horizontal_indices = torch.where(torch.any(mask, axis=0))[0]
    vertical_indices = torch.where(torch.any(mask, axis=1))[0]
    if horizontal_indices.shape[0]:
      x1, x2 = horizontal_indices[[0, -1]]
      y1, y2 = vertical_indices[[0, -1]]

      x1 -= 1
      y1 -= 1
      # x2 and y2 should not be part of the box. Increment by 1.
      x2 += 1
      y2 += 1
    else:
      # No mask for this instance. Might happen due to
      # resizing or cropping. Set bbox to zeros
      x1, x2, y1, y2 = 0, 0, 0, 0

    # normalize coords
    height, width = mask.shape[-2:]
    x1 = x1 / width
    x2 = x2 / width
    y1 = y1 / height
    y2 = y2 / height

    return torch.Tensor([x1, y1, x2, y2])

  def __len__(self):
    return self.wrapped_dataset.__len__()

  def __getitem__(self, idx):
    (input, label) = self.wrapped_dataset.__getitem__(idx)
    input += 0.5
    bbox = BBOXDataset._get_bounding_box(label)
    return input, {'boxes': bbox.unsqueeze(0), 'labels': torch.ones((1,), dtype=torch.int64)}

