'''
Loads all of the scans from the scans/ folder and saves each slice as a separate
numpy array file into the appropriate folder (e.g. train/<patient number>-<slice number>.npy) 
in the currrent directory.
'''

import os.path as p
import sys

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import cv2 as cv

sys.path.append('../..')
import helpers as h

def read_scan(file_path):
  ''' Read scan with axial view '''
  scan = np.rot90(nib.load(file_path).get_fdata())
  return scan

scans_directory = 'scans/'
subfolders = ['train', 'test', 'valid']

for folder in subfolders:
  all_files = h.listdir(p.join(scans_directory, folder))
  all_files = np.array(all_files)
  all_files.sort()

  file_len = len(all_files) // 2
  all_files = np.dstack((all_files[file_len:], all_files[:file_len])).squeeze()

  h.mkdir(folder)

  for scan_files in all_files:
    volume_file, mask_file = scan_files[0], scan_files[1]
    
    volume_scan = read_scan(p.join(scans_directory, folder, volume_file))
    mask_scan = read_scan(p.join(scans_directory, folder, mask_file))

    for i in range(mask_scan.shape[-1]):
      mask_slice = mask_scan[..., i]
      if mask_slice.sum() <= 0:
        # skip non-liver slices
        continue

      volume_slice = volume_scan[..., i]
      original_volume_name = volume_file.split('.')[0]
      original_mask_name = mask_file.split('.')[0]

      volume_name = f'{original_volume_name}-{i}.npy'
      mask_name = f'{original_mask_name}-{i}.npy'

      volume_save_path = p.join(folder, volume_name)
      mask_save_path = p.join(folder, mask_name)

      volume_slice = cv.resize(volume_slice, dsize=(128, 128), interpolation=cv.INTER_CUBIC)
      mask_slice = cv.resize(mask_slice, dsize=(128, 128), interpolation=cv.INTER_CUBIC)

      np.save(volume_save_path, volume_slice)
      np.save(mask_save_path, mask_slice)

