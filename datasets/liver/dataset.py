import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

sys.path.append('..')
import helpers as h
import polar_transformations

NUM_SLICES_PER_SCAN = 841
WINDOW_MAX = 200
WINDOW_MIN = 0
# obtained empirically
GOLBAL_PIXEL_MEAN = 0.1

class LiverDataset(Dataset):

  def read_scan(self, file_path):
    scan = np.rot90(nib.load(file_path).get_fdata())
    return scan

  def __init__(self, directory):
    all_files = h.listdir(directory)
    all_files = np.array(all_files)
    all_files.sort()

    self.directory = directory
    self.data = np.dstack((all_files[len(all_files) // 2:], all_files[:len(all_files) // 2])).squeeze()
    
  def __len__(self):
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
    volume_slice -= GOLBAL_PIXEL_MEAN

    # convert to polar
    center = polar_transformations.centroid(mask_slice)
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




  
