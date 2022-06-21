import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

import albumentations as A

sys.path.append('..')
import helpers as h
import polar_transformations

WINDOW_MAX = 500
WINDOW_MIN = 200
# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

class AortaDataset(Dataset):

  in_channels = 3
  out_channels = 3

  height = 256
  width = 256

  @staticmethod
  def get_augmentation():
    transform = A.Compose([
      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=.5),
      A.HorizontalFlip(p=0.3),
    ], keypoint_params=A.KeypointParams(format='xy'))
    return transform

  @staticmethod
  def get_patient_names(hospital_id):
    all_files = h.listdir('datasets/aa/label')
    all_slices = [f.split('-')[0] for f in all_files]
    patient_names = list(set(all_slices))
    patient_names = [s for s in patient_names if hospital_id in s]
    patient_names.sort()
    return patient_names

  def __init__(self, mode, polar=True, manual_centers=None, patient_names=None, depth=False, center_augmentation=False, percent=None, hospital_id='D'):
    '''
    manual_centers: an array of centers and their file names in the form of
    [(tuple, str)], for example:
    [([150, 200], 'D0-1.npy'), ([300, 100], 'D0-1.npy'), ...]
    depth: use 3 consequtive slices for training and inference
    hospital_id: one of 'D', 'K' or 'R'
    '''
    AortaDataset.in_channels = 3 if depth else 1
    AortaDataset.out_channels = 3 if depth else 1
    self.directory = 'datasets/aa'
    self.polar = polar
    self.depth = depth
    self.center_augmentation = center_augmentation

    print(patient_names)

    all_files = h.listdir(p.join(self.directory, 'label'))
    all_files = np.array(all_files)
    all_files.sort()
    all_files = [f for f in all_files if hospital_id in f]
    
    if patient_names is not None:
      patient_files = []
      for name in patient_names:
        patient_files += [f for f in all_files if name in f]
      all_files = patient_files

    self.file_names = all_files

    if mode == 'train':
      self.transform = AortaDataset.get_augmentation()
    else:
      self.transform = None

    if self.polar:
      self.centers = self.get_centers() if manual_centers is None else manual_centers
      if patient_names is not None:
        filtered_centers = []
        for name in patient_names:
          filtered_centers += [(c, f) for (c, f) in self.centers if name in f]
        self.centers = filtered_centers
    else:
      self.centers = None

  def get_centers(self):
    '''
    Returns an array of centers and their file names in the form of
    [(tuple, str)], for example:
    [([150, 200], 'D0-1.npy'), ([300, 100], 'D0-1.npy'), ...]
    '''
    centers = []
    for file_name in self.file_names:
      mask_slice = np.load(p.join(self.directory, 'label', file_name))
      centers_for_file = polar_transformations.centroids(mask_slice)
      mask_slice = None
      del mask_slice

      for center in centers_for_file:
        if self.center_augmentation and np.random.uniform() < 0.3:
          center_max_shift = 0.01 * 256
          center = (
            center[0] + np.random.uniform(-center_max_shift, center_max_shift),
            center[1] + np.random.uniform(-center_max_shift, center_max_shift))

        centers.append((center, file_name))
    
    return centers
    
  def __len__(self):
    # return 16 # overfit single batch
    return len(self.centers) if self.polar else len(self.file_names)

  def __getitem__(self, idx):
    if self.polar:
      current_slice_file = self.centers[idx][1]
    else:
      current_slice_file = self.file_names[idx]

    if self.depth:
      # file name structure: <scan>-<slice-number>.npy, e.g. D2-2.npy
      slice_number = int(current_slice_file.split('-')[-1].split('.')[0])
      scan_name = current_slice_file.split('-')[0]
      prev_slice_file = f'{scan_name}-{slice_number - 1}.npy'
      next_slice_file = f'{scan_name}-{slice_number + 1}.npy'
      slice_files = [prev_slice_file, current_slice_file, next_slice_file]

      scan = np.zeros((3, AortaDataset.width, AortaDataset.height), dtype=np.int16)
      mask = np.zeros((3, AortaDataset.width, AortaDataset.height), dtype=np.int16)

      for (slice_index, slice_file) in enumerate(slice_files):
        if not p.exists(p.join(self.directory, 'input', slice_file)):
          continue

        volume_slice = np.load(p.join(self.directory, 'input', slice_file))
        scan[slice_index] = volume_slice.copy()
        mask_slice = np.load(p.join(self.directory, 'label', slice_file))
        mask[slice_index] = mask_slice.copy()

        volume_slice = None
        mask_slice = None
        del volume_slice
        del mask_slice
    else:
      scan = np.load(p.join(self.directory, 'input', current_slice_file))
      scan = np.expand_dims(scan, axis=0)
      mask = np.load(p.join(self.directory, 'label', current_slice_file))
      mask = np.expand_dims(mask, axis=0)

    # window input slice
    scan[scan > WINDOW_MAX] = WINDOW_MAX
    scan[scan < WINDOW_MIN] = WINDOW_MIN

    scan = scan.astype(np.float64)
    
    # normalize and zero-center
    scan = (scan - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    # zero-centered globally because CT machines are calibrated to have even 
    # intensities across images
    scan -= GLOBAL_PIXEL_MEAN

    if self.polar:
      center = self.centers[idx][0]

    if self.transform is not None:
      transformed = self.transform(image=scan[0], mask=mask[0], keypoints=[center] if self.polar else [])
      if self.polar:
        center = transformed['keypoints'][0]
      mask[0] = transformed['mask']
      scan[0] = transformed['image']

    # convert to polar
    if self.polar:
      for i in range(scan.shape[0]):
        scan[i] = polar_transformations.to_polar(scan[i], center)
        mask[i] = polar_transformations.to_polar(mask[i], center)

    volume_tensor = torch.from_numpy(scan).float()
    mask_tensor = torch.from_numpy(mask).float()

    scan = None
    mask = None
    del scan
    del mask

    return volume_tensor, mask_tensor
