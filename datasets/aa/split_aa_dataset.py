'''
Loads all of the scans from the data/ folder and saves each slice as a separate
numpy array file into the appropriate folder (e.g. train/<patient number>-<slice number>.npy) 
in the currrent directory.
'''

import os
import os.path as p
import sys

import matplotlib.pyplot as plt
import numpy as np
import nrrd
import cv2 as cv

sys.path.append('../..')
import helpers as h

def read_scan(file_path):
  ''' Read scan with axial view '''
  data, _ = nrrd.read(file_path)
  scan = np.rot90(data)
  scan = scan.astype(np.int16)
  return scan

scans_directory = 'data/avt/'

all_files = h.listdir(scans_directory)
all_files.sort()
label_files = [f for f in all_files if 'seg' in f]
np.random.seed(42)
np.random.shuffle(label_files)
print(label_files)

h.mkdir('input')
h.mkdir('label')

for mask_file in label_files:
  volume_file = mask_file.replace('seg.', '')
  volume_scan = read_scan(p.join(scans_directory, volume_file))
  mask_scan = read_scan(p.join(scans_directory, mask_file))

  for i in range(mask_scan.shape[-1]):
    mask_slice = mask_scan[..., i]
    if mask_slice.sum() <= 0:
      # skip empty slices
      continue

    volume_slice = volume_scan[..., i]
    original_mask_name = mask_file.split('.')[0]

    volume_name = f'{original_mask_name}-{i}.npy'
    mask_name = f'{original_mask_name}-{i}.npy'

    volume_save_path = p.join('input', volume_name)
    mask_save_path = p.join('label', mask_name)
    print(volume_name, volume_slice.dtype)

    volume_slice = cv.resize(volume_slice, dsize=(256, 256), interpolation=cv.INTER_CUBIC)
    mask_slice = cv.resize(mask_slice, dsize=(256, 256), interpolation=cv.INTER_CUBIC)

    np.save(volume_save_path, volume_slice)
    np.save(mask_save_path, mask_slice)

