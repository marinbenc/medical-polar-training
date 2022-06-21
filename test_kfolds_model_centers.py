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
import scipy

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

  logs = f'{args.non_polar_experiment}'
  folds = os.listdir(logs)
  folds.sort()

  metrics_per_patient = {}
  for patient in patients:
    metrics_per_patient[patient] = []


  for fold, (_, valid_idxs) in zip(folds, splits):
    non_polar_weights_folder = os.path.join(args.non_polar_experiment, fold)
    non_polar_weights = [f for f in os.listdir(non_polar_weights_folder) if 'best_model' in f][-1]

    # find centroids
    non_polar_dataset = dataset_class('test', polar=False, depth=args.depth, patient_names=patients[valid_idxs])
    non_polar_model = get_model(os.path.join(non_polar_weights_folder, non_polar_weights), dataset_class, device, args)
    xs, non_polar_ys, non_polar_predictions = get_predictions(non_polar_model, non_polar_dataset, device)
    
    centers = []
    for i in range(len(non_polar_dataset)):
      file_name = non_polar_dataset.file_names[i]
      if args.depth:
        file_centers = polar_transformations.centroids(non_polar_predictions[i][1])
      else:
        file_centers = polar_transformations.centroids(non_polar_predictions[i])

      for center in file_centers:
        centers.append((center, file_name))

    # run final predictions

    polar_weights_folder = os.path.join(args.polar_experiment, fold)
    polar_weights = [f for f in os.listdir(polar_weights_folder) if 'best_model' in f][-1]

    polar_dataset = dataset_class('test', polar=True, manual_centers=centers, depth=args.depth, patient_names=patients[valid_idxs])
    polar_model = get_model(os.path.join(polar_weights_folder, polar_weights), dataset_class, device, args)
    _, all_ys, all_predicted_ys = get_predictions(polar_model, polar_dataset, device)

    processed_ys = []
    processed_predicted_ys = []
    processed_filenames = []
    processed_xs = []
    processed_np_preds = []

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
        results[file_name]['np_pred'] = non_polar_predictions[idx_in_non_polar]
        results[file_name]['input'] = xs[idx_in_non_polar]

    for file_name in results.keys():
      viz_images = []

      preds = results[file_name]['preds']
      gt_center = results[file_name]['gt'][0]
      if len(preds) == 1:
        center, pred = preds[0]
        pred = polar_transformations.to_cart(pred, center)
        pred = polar_transformations.to_polar(pred, gt_center)
      else:
        viz_images.append(results[file_name]['input'])
        viz_images.append(results[file_name]['np_pred'])
        weighted_slices = np.zeros((len(preds), *preds[0][1].shape), dtype=preds[0][1].dtype)
        for i in range(len(preds)):
          center, pred = preds[i]
          pred[pred < 0.5] = 0
          pred[pred >= 0.5] = 1

          cc = polar_transformations.lcc(pred)
          pred += cc
          viz_images.append(pred.copy())
          pred = polar_transformations.to_cart(pred, center)
          viz_images.append(pred.copy())
          pred = polar_transformations.to_polar(pred, gt_center)
          weighted_slices[i] = pred.copy()

        processed_ys.append(polar_transformations.to_cart((results[file_name]['gt'][1] * 255).astype(np.uint8), gt_center))

        weighted_slices = weighted_slices.sum(axis=0)
        weighted_slices /= weighted_slices.max()
        viz_images.append(polar_transformations.to_cart(weighted_slices, gt_center))

        weighted_slices = filters.apply_hysteresis_threshold(weighted_slices, 0, args.hist_th)

        viz_images.append(polar_transformations.to_cart(weighted_slices.astype(np.uint8) * 255, gt_center))
        viz_images.append(polar_transformations.to_cart(results[file_name]['gt'][1], gt_center))

        processed_filenames.append(file_name) 
        # TODO Remove cart transform
        processed_predicted_ys.append(polar_transformations.to_cart((weighted_slices * 255).astype(np.uint8), gt_center))
        processed_xs.append(np.load(os.path.join('datasets/aa/input', file_name)))
        processed_np_preds.append(results[file_name]['np_pred'])

    for (i, file_name) in enumerate(processed_filenames):
      file_dsc = dsc(processed_predicted_ys[i], processed_ys[i])
      file_iou = iou(processed_predicted_ys[i], processed_ys[i])
      file_prec = precision(processed_predicted_ys[i], processed_ys[i])
      file_rec = recall(processed_predicted_ys[i], processed_ys[i])
      metrics = (file_dsc, file_iou, file_prec, file_rec)

      for patient in patients:
        if patient in file_name:
          metrics_per_patient[patient].append(np.array(metrics))

    # while True:
    #   indices = np.random.randint(0, high=len(processed_predicted_ys), size=5)
    #   initial_predictions = np.array(processed_np_preds)[indices]
    #   predictions = np.array(processed_predicted_ys)[indices]
    #   gts = np.array(processed_ys)[indices]
    #   xs = np.array(processed_xs)[indices]
    #   viz_images = list(sum(zip(xs.tolist(), initial_predictions.tolist(), predictions.tolist(), gts.tolist()), ()))
    #   size = 128
    #   viz_images = [np.array(img)[256//2-size//2:256//2+size//2, 256//2-size//2:256//2+size//2] for img in viz_images]
    #   helpers.show_images_row(viz_images, rows=len(indices), figsize=(6, 16), cmap='gray')
    #   plt.tight_layout()
    #   plt.show()

  all_metrics = []
  for patient in patients:
    metrics = np.array(metrics_per_patient[patient])
    if len(metrics) > 0:
      all_metrics.append(metrics.mean(axis=0))

  all_metrics = np.array(all_metrics)
  np.save(f'results_test_kfolds_model_centers_{args.polar_experiment.split("/")[-1]}.npy', all_metrics)
  print(all_metrics.mean(axis=0))
  print(all_metrics.std(axis=0))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Testing'
  )
  parser.add_argument(
    '--non-polar-experiment', type=str, help='path to weights of non-polar model'
  )
  parser.add_argument(
    '--polar-experiment', type=str, help='path to weights of polar model'
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
  parser.add_argument(
    '--folds',
    type=int,
    default=4,
    help='k in k-folds cross-validation',
  )
  args = parser.parse_args()
  main(args)

