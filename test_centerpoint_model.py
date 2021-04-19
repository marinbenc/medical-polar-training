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
import cv2 as cv

sys.path.append('datasets/liver')
from liver_dataset import LiverDataset
sys.path.append('datasets/polyp')
from polyp_dataset import PolypDataset

import train_hourglass
from heatmap_dataset import HeatmapDataset

import train
from helpers import dsc, iou, precision, recall
import polar_transformations
from test import get_predictions

from helpers import show_images_row

def get_centerpoint_predictions(model, dataset, device):
  all_xs = []
  all_ys = []
  all_predicted_ys = []

  with torch.no_grad():
    for (x, y) in dataset:
      x = x.to(device)
      prediction = model(x.unsqueeze(0).detach())

      all_xs.append(x.squeeze().detach().cpu().numpy().transpose(1, 2, 0) + 0.5)

      predicted_y = prediction
      predicted_y = predicted_y.squeeze().detach().cpu().numpy()

      all_predicted_ys.append(predicted_y[-1])

      y = y.squeeze().detach().cpu().numpy()
      all_ys.append(y)

  # N = 4

  # centers = [cv.minMaxLoc(cv.resize(center, all_xs[0].shape[:2]))[-1] for center in all_predicted_ys[:N]]
  # xs = [cv.circle(x, center, 7, (0, 0, 255), -1) for (x, center) in zip(all_xs[:N], centers)]

  # show_images_row(xs + all_ys[:N], titles=["Input" for _ in range(N)] + ["Heatmap" for _ in range(N)], rows=2)
  # plt.show()
  
  return all_ys, all_predicted_ys

def get_centerpoint_model(weights, dataset_class, device):
  print(dataset_class)
  model = train_hourglass.get_model(args, dataset_class, device)
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
  dataset_class = train.get_dataset_class(args)

  # find centroids
  centers_dataset = HeatmapDataset(args.dataset, 'test')
  centers_model = get_centerpoint_model(args.centerpoint_weights, dataset_class, device)
  centers_gt, centers_pred = get_centerpoint_predictions(centers_model, centers_dataset, device)
  centers = [cv.minMaxLoc(cv.resize(center, (dataset_class.width, dataset_class.height)))[-1] for center in centers_pred]

  # test_dataset = dataset_class('test', polar=False)
  # for i in range(8):
  #   plt.imshow(centers_pred[i][-1])
  #   #plt.imshow(test_dataset[i][1].detach().cpu().numpy().squeeze())
  #   #plt.scatter(centers[i][0], centers[i][1])
  #   plt.show()

  # run final predictions
  polar_dataset = dataset_class('test', polar=True, manual_centers=centers)
  polar_model = get_model(args.polar_weights, dataset_class, device)
  _, all_ys, all_predicted_ys = get_predictions(polar_model, polar_dataset, device)

  dscs = np.array([dsc(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  ious = np.array([iou(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  precisions = np.array([precision(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  recalls = np.array([recall(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])

  # sorting = np.argsort(dscs)
  # sorted_final = np.array(all_ys)[sorting]
  # sorted_ys = np.array(centers_gt)[sorting]
  # sorted_non_polar_predictions = np.array(centers_pred)[sorting]
  # sorted_centers = np.array(centers)[sorting]

  # for i in range(len(sorting)):
  #   plt.imshow(sorted_final[i])
  #   c = sorted_centers[i]
  #   plt.scatter(c[0], c[1])
  #   plt.show()

  print(f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Testing'
  )
  parser.add_argument(
    '--centerpoint_weights', type=str, help='path to weights of centerpoint model'
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
  parser.add_argument(
      '--nstacks',
      type=int,
      default=8,
      help='number of hourglass stacks',
  )

  args = parser.parse_args()
  main(args)

