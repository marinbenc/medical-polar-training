import sys
import os.path as p
import torch
from torch.utils.data import random_split
from shutil import copy

sys.path.append('../..')
import helpers as h

dataset_folder = 'CVC-ClinicDB/'
gt_folder = 'CVC-ClinicDB/Ground Truth/'
input_folder = 'CVC-ClinicDB/Original/'

gt_files = h.listdir(p.join(dataset_folder, 'Ground Truth'))
gt_files.sort()

# 0.8, 0.1, 0.1
train, test, valid = (490, 61, 61)
# sanity check
assert(train + test + valid == len(gt_files))

split = random_split(
  gt_files, (train, test, valid),
  generator=torch.Generator().manual_seed(42))

folder_names = ['train', 'test', 'valid']

for i in range(len(folder_names)):
  folder = folder_names[i]
  split_label_folder = p.join(folder, 'label')
  h.mkdir(split_label_folder)
  split_input_folder = p.join(folder, 'input')
  h.mkdir(split_input_folder)

  files = split[i]
  for file_name in files:
    gt_file = p.join(gt_folder, file_name)
    gt_dst = p.join(split_label_folder, file_name)
    copy(gt_file, gt_dst)

    input_file = p.join(input_folder, file_name)
    input_dst = p.join(split_input_folder, file_name)
    copy(input_file, input_dst)
