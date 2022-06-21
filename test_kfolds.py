'''
Tests the datasets and reports mean DSC.
The first model is used to obtain a center for transforming
the data into polar coordinates. The second model then performs
a final prediction.
'''

import argparse
from concurrent.futures import process
import sys
import os.path as p
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append('datasets/liver')
from liver_dataset import LiverDataset
sys.path.append('datasets/polyp')
from polyp_dataset import PolypDataset

from scipy.ndimage.measurements import label
from skimage import filters

import train
import helpers
from helpers import dsc, iou, precision, recall
import polar_transformations
from test import get_predictions

def get_model(weights, dataset_class, device, args):
  model = train.get_model(args, dataset_class, device)
  model.to(device)
  model.load_state_dict(torch.load(weights))
  model.eval()
  model.train(False)
  return model

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

  dataset_class = train.get_dataset_class(args)
  patients, splits = train.get_splits(args, dataset_class)

  logs = f'{args.experiment}'
  folds = os.listdir(logs)
  folds.sort()

  metrics_per_patient = {}
  for patient in patients:
    metrics_per_patient[patient] = []


  for fold, (_, valid_idxs) in zip(folds, splits):
    weights_folder = os.path.join(args.experiment, fold)
    weights = [f for f in os.listdir(weights_folder) if 'best_model' in f][-1]

    # find centroids
    dataset = dataset_class('test', polar=args.polar, depth=args.depth, patient_names=patients[valid_idxs])
    model = get_model(os.path.join(weights_folder, weights), dataset_class, device, args)
    _, ys, predictions = get_predictions(model, dataset, device)

    file_names = []
    if args.polar:
      centers = dataset.get_centers()
      for (_, file_name) in centers:
        file_names.append(file_name)
    else:
      file_names = dataset.file_names
    
    for (i, file_name) in enumerate(file_names):
      file_dsc = dsc(ys[i], predictions[i])
      file_iou = iou(ys[i], predictions[i])
      file_prec = precision(ys[i], predictions[i])
      file_rec = recall(ys[i], predictions[i])
      metrics = (file_dsc, file_iou, file_prec, file_rec)

      for patient in patients:
        if patient in file_name:
          metrics_per_patient[patient].append(np.array(metrics))

  all_metrics = []
  for patient in patients:
    metrics = np.array(metrics_per_patient[patient])
    if len(metrics) > 0:
      all_metrics.append(metrics.mean(axis=0))

  all_metrics = np.array(all_metrics)
  np.save(f'results_test_kfolds_{args.experiment.split("/")[-1]}.npy', all_metrics)
  print(all_metrics.mean(axis=0))
  print(all_metrics.std(axis=0))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Testing'
  )
  parser.add_argument(
    '--experiment', type=str, help='path to weights of model'
  )
  parser.add_argument(
    '--model', type=str, choices=train.model_choices, default='liver', help='dataset type'
  )
  parser.add_argument(
    '--dataset', type=str, choices=train.dataset_choices, default='liver', help='dataset type'
  )
  parser.add_argument(
    '--hist-th', type=float
  )
  parser.add_argument(
      '--depth', 
      action='store_true',
      help='use 3 consequtive slices')
  parser.add_argument(
      '--polar', 
      action='store_true',
      help='use polar coords')
  parser.add_argument(
      '--hospital-id', type=str, choices=['D', 'K', 'R'], default='D', help='which dataset subset (center) to use'
  )
  parser.add_argument(
    '--folds',
    type=int,
    default=4,
    help='k in k-folds cross-validation',
  )
  args = parser.parse_args()
  main(args)

