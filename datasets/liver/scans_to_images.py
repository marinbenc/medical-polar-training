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
  all_files.sort()

  h.mkdir(folder)

  for scan_file in all_files:
    scan = read_scan(p.join(scans_directory, folder, scan_file))
    for i in range(scan.shape[-1]):
      current_slice = scan[..., i]
      original_name = scan_file.split('.')[0]
      file_name = f'{original_name}-{i}.npy'
      save_path = p.join(folder, file_name)
      np.save(save_path, current_slice)


