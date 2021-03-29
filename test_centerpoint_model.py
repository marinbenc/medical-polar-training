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

sys.path.append('models')
from centerpoint_model import centerpoint_model

import train
from helpers import dsc, iou, precision, recall
import polar_transformations
from test import get_predictions

def get_centerpoint_predictions(model, dataset, device):
  all_ys = []
  all_predicted_ys = []

  with torch.no_grad():
    for (x, y) in dataset:
      x = x.to(device)
      prediction = model(x.unsqueeze(0).detach())

      predicted_y = prediction
      predicted_y = tuple(predicted_y.squeeze().detach().cpu().numpy())

      all_predicted_ys.append(predicted_y)

      y = tuple(y.squeeze().detach().cpu().numpy())
      all_ys.append(y)


      print(y, predicted_y)

      # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
      # ax1.imshow(y)
      # ax1.set_title('GT')
      # ax2.imshow(predicted_y.squeeze())
      # ax2.set_title('Predicted')
      # plt.show()


  return all_ys, all_predicted_ys

def get_centerpoint_model(weights, device):
  model = centerpoint_model()
  model.to(device)
  model.load_state_dict(torch.load(weights))
  model.eval()
  model.train(False)
  return model

def get_model(weights, dataset_class, device):
  args.loss = 'dice'
  model = train.get_model(args, dataset_class, device)
  model.to(device)
  model.load_state_dict(torch.load(weights))
  model.eval()
  model.train(False)
  return model

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

  if args.dataset == 'liver':
    dataset_class = LiverDataset
  elif args.dataset == 'polyp':
    dataset_class = PolypDataset
    
  # find centroids
  centers_dataset = dataset_class('test', polar=False, centers=True)
  centers_model = get_centerpoint_model(args.non_polar_weights, device)
  _, centers = get_centerpoint_predictions(centers_model, centers_dataset, device)

  # run final predictions
  polar_dataset = dataset_class('test', polar=True, manual_centers=centers)
  polar_model = get_model(args.polar_weights, dataset_class, device)
  _, all_ys, all_predicted_ys = get_predictions(polar_model, polar_dataset, device)

  dscs = np.array([dsc(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  ious = np.array([iou(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  precisions = np.array([precision(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  recalls = np.array([precision(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])

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

