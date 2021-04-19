'''
Tests the datasets and reports mean DSC.
The first model is used to obtain a center for transforming
the data into polar coordinates. The second model then performs
a final prediction.
'''

import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append('datasets/liver')
from liver_dataset import LiverDataset
sys.path.append('datasets/polyp')
from polyp_dataset import PolypDataset

import train
from helpers import dsc, iou, precision, recall
import polar_transformations
from test import get_predictions

def get_model(weights, dataset_class, device):
  model = train.get_model(args, dataset_class, device)
  model.to(device)
  model.load_state_dict(torch.load(weights))
  model.eval()
  model.train(False)
  return model

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  dataset_class = train.get_dataset_class(args)
    
  # find centroids
  non_polar_dataset = dataset_class('test', polar=False)
  non_polar_model = get_model(args.non_polar_weights, dataset_class, device)
  _, non_polar_ys, non_polar_predictions = get_predictions(non_polar_model, non_polar_dataset, device)
  centers = [polar_transformations.centroid(prediction) for prediction in non_polar_predictions]

  # run final predictions
  polar_dataset = dataset_class('test', polar=True, manual_centers=centers)
  polar_model = get_model(args.polar_weights, dataset_class, device)
  _, all_ys, all_predicted_ys = get_predictions(polar_model, polar_dataset, device)

  dscs = np.array([dsc(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  ious = np.array([iou(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  precisions = np.array([precision(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  recalls = np.array([recall(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])


  print(f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Testing'
  )
  parser.add_argument(
    '--non_polar_weights', type=str, help='path to weights of non-polar model'
  )
  parser.add_argument(
    '--polar_weights', type=str, help='path to weights of polar model'
  )
  parser.add_argument(
    '--model', type=str, choices=train.model_choices, default='liver', help='dataset type'
  )
  parser.add_argument(
    '--dataset', type=str, choices=train.dataset_choices, default='liver', help='dataset type'
  )

  args = parser.parse_args()
  main(args)

