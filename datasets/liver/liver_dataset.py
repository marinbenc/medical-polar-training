import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

sys.path.append('..')
import helpers as h
import polar_transformations

NUM_SLICES_PER_SCAN = 841
WINDOW_MAX = 200
WINDOW_MIN = 0
# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

class LiverDataset(Dataset):

  in_channels = 1
  out_channels = 1

  height = 128
  width = 128

  def __init__(self, directory, polar=True, manual_centers=None, center_augmentation=False):
    self.directory = p.join('datasets/liver', directory)
    self.polar = polar
    self.manual_centers = manual_centers
    self.center_augmentation = center_augmentation

    all_files = h.listdir(self.directory)
    all_files = np.array(all_files)
    all_files.sort()

    self.data = np.dstack((all_files[len(all_files) // 2:], all_files[:len(all_files) // 2])).squeeze()
    self.file_names = list(all_files[len(all_files) // 2:])
    
  def __len__(self):
    # return 16 # overfit single batch
    return len(self.data)

  def __getitem__(self, idx):
    current_data = self.data[idx]
    volume, mask = current_data[0], current_data[1]
    volume_slice = np.load(p.join(self.directory, volume))
    mask_slice = np.load(p.join(self.directory, mask))

    # remove non-liver labels
    mask_slice[mask_slice > 1] = 1

    # window input slice
    volume_slice[volume_slice > WINDOW_MAX] = WINDOW_MAX
    volume_slice[volume_slice < WINDOW_MIN] = WINDOW_MIN
    
    # normalize and zero-center
    volume_slice = (volume_slice - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    # zero-centered globally because CT machines are calibrated to have even 
    # intensities across images
    volume_slice -= GLOBAL_PIXEL_MEAN

    # convert to polar
    if self.polar:
      if self.manual_centers is not None:
        center = self.manual_centers[idx]
      else:
        center = polar_transformations.centroid(mask_slice)

      if self.center_augmentation and np.random.uniform() < 0.3:
        center_max_shift = 0.05 * LiverDataset.height
        center = np.array(center)
        center = (
          center[0] + np.random.uniform(-center_max_shift, center_max_shift),
          center[1] + np.random.uniform(-center_max_shift, center_max_shift))

      volume_slice = polar_transformations.to_polar(volume_slice, center)
      mask_slice = polar_transformations.to_polar(mask_slice, center)

    # convert to Pytorch expected format
    volume_slice = np.expand_dims(volume_slice, axis=-1)
    volume_slice = volume_slice.transpose(2, 0, 1)
    mask_slice = np.expand_dims(mask_slice, axis=-1)
    mask_slice = mask_slice.transpose(2, 0, 1)

    volume_tensor = torch.from_numpy(volume_slice.astype(np.float32))
    mask_tensor = torch.from_numpy(mask_slice.astype(np.float32))

    return volume_tensor, mask_tensor




  
