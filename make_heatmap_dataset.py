'''
Creates a heatmap/ folder inside the dataset folder with the correctly sized input and heatmap images.
'''

import argparse
import os.path as p

import numpy as np
import albumentations.augmentations.functional as F
import cv2 as cv

import polar_transformations
import helpers as h
from train import dataset_choices, get_dataset_class

# based on https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/data/MPII/dp.py
def generate_heatmap(output_res, center):
  sigma = max(output_res) / 8
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

def process(input, label):
  input = input.detach().numpy().transpose(1, 2, 0)
  input = F.resize(input, 256, 256)
  label = label.detach().numpy().squeeze()
  label = F.resize(label, 256, 256)
  
  center = polar_transformations.centroid(label)

  label = generate_heatmap(label.shape[-2:], center)
  label = F.resize(label, 64, 64)

  return input, label

def save(input, label, file_name, folder, args):
  dataset_folder = p.join('datasets', args.dataset, 'heatmap')
  h.mkdir(dataset_folder)
  save_folder = p.join(dataset_folder, folder)
  input_folder = p.join(save_folder, 'input')
  h.mkdir(input_folder)
  label_folder = p.join(save_folder, 'label')
  h.mkdir(label_folder)

  np.save(p.join(input_folder, file_name), input)
  np.save(p.join(label_folder, file_name), label)

def main(args):
  original_dataset_class = get_dataset_class(args)
  folders = ['train', 'test', 'valid']
  datasets = [original_dataset_class(directory=folder, polar=False) for folder in folders]

  for folder, dataset in zip(folders, datasets):
    for i, (input, label) in enumerate(dataset):
      input, label = process(input, label)
      file_name = dataset.file_names[i] + '.npy'
      save(input, label, file_name, folder, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a heatmap/ folder inside the dataset folder with the correctly sized input and heatmap images.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=dataset_choices, default='liver', help='which dataset to use'
    )
    args = parser.parse_args()
    main(args)