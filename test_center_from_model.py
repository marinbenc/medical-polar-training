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
    
  # find centroids
  non_polar_dataset = dataset_class('test', polar=False, depth=args.depth)
  non_polar_model = get_model(args.non_polar_weights, dataset_class, device, args)
  _, non_polar_ys, non_polar_predictions = get_predictions(non_polar_model, non_polar_dataset, device)
  
  centers = []
  for i in range(len(non_polar_dataset)):
    file_name = non_polar_dataset.file_names[i]
    if args.depth:
      file_centers = polar_transformations.centroids(non_polar_predictions[i][1])
    else:
      file_centers = polar_transformations.centroids(non_polar_predictions[i])

      # if len(gt_centers) != len(file_centers):
      #   helpers.show_images_row([non_polar_ys[i], non_polar_predictions[i]])
      #   plt.show()
    for center in file_centers:
      centers.append((center, file_name))

  # run final predictions
  polar_dataset = dataset_class('test', polar=True, manual_centers=centers, depth=args.depth)
  polar_model = get_model(args.polar_weights, dataset_class, device, args)
  _, all_ys, all_predicted_ys = get_predictions(polar_model, polar_dataset, device)

  processed_ys = []
  processed_predicted_ys = []
  processed_filenames = []

  results = {}

  for i in range(len(centers)):
    predicted_y = all_predicted_ys[i]
    if args.depth:
      predicted_y = predicted_y[1]
    center, file_name = centers[i]
    if file_name in results:
      results[file_name]['preds'].append((center, predicted_y))
    else:
      results[file_name] = {}
      results[file_name]['preds'] = [(center, predicted_y)]
      idx_in_non_polar = non_polar_dataset.file_names.index(file_name)
      y = all_ys[i]
      if args.depth:
        y = y[1]
      results[file_name]['gt'] = (center, y)
      results[file_name]['np_pred'] = non_polar_predictions[i]

  kernel = np.ones((4,4),np.uint8)

  for file_name in results.keys():
    viz_images = []

    preds = results[file_name]['preds']
    gt_center = results[file_name]['gt'][0]
    if len(preds) == 1:
      center, pred = preds[0]
      pred = polar_transformations.to_cart(pred, center)
      pred = polar_transformations.to_polar(pred, gt_center)
    else:
      viz_images.append(results[file_name]['np_pred'])
      weighted_slices = np.zeros((len(preds), *preds[0][1].shape), dtype=preds[0][1].dtype)
      for i in range(len(preds)):
        center, pred = preds[i]
        #plt.imshow(pred)
        #plt.show()
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1

        cc = polar_transformations.lcc(pred)
        pred += cc

        viz_images.append(pred.copy())
        pred = polar_transformations.to_cart(pred, center)
        viz_images.append(pred.copy())
        pred = polar_transformations.to_polar(pred, gt_center)
        weighted_slices[i] = pred.copy()

      processed_ys.append(results[file_name]['gt'][1])

      weighted_slices = weighted_slices.sum(axis=0)
      weighted_slices /= weighted_slices.max()
      viz_images.append(polar_transformations.to_cart(weighted_slices, gt_center))
      weighted_slices = filters.apply_hysteresis_threshold(weighted_slices, 0, args.hist_th)
      viz_images.append(polar_transformations.to_cart(weighted_slices, gt_center))
      viz_images.append(polar_transformations.to_cart(results[file_name]['gt'][1], gt_center))
      #weighted_slices[weighted_slices > 0] = 1
      #helpers.show_images_row([weighted_slices, results[file_name]['gt']])
      #plt.show()

      #plt.imshow(results[file_name]['gt'] + weighted_slices)
      #plt.show()

      processed_filenames.append(file_name)
      processed_predicted_ys.append(weighted_slices)

  dscs = np.array([dsc(processed_predicted_ys[i], processed_ys[i]) for i in range(len(processed_predicted_ys))])
  ious = np.array([iou(processed_predicted_ys[i], processed_ys[i]) for i in range(len(processed_predicted_ys))])
  precisions = np.array([precision(processed_predicted_ys[i], processed_ys[i]) for i in range(len(processed_predicted_ys))])
  recalls = np.array([recall(processed_predicted_ys[i], processed_ys[i]) for i in range(len(processed_predicted_ys))])

  print(f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')
  return dscs.mean(), ious.mean(), precisions.mean(), recalls.mean()

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
  parser.add_argument(
    '--hist-th', type=float
  )
  parser.add_argument(
      '--depth', 
      action='store_true',
      help='use 3 consequtive slices')

  parser.add_argument(
      '--hospital-id', type=str, choices=['D', 'K', 'R'], default='D', help='which dataset subset (center) to use'
  )
  args = parser.parse_args()
  main(args)

