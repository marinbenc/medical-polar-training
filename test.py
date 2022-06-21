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
from train import get_model
from helpers import dsc, iou, precision, recall, show_images_row
import polar_transformations

def get_predictions(model, dataset, device):
  all_xs = []
  all_ys = []
  all_predicted_ys = []

  with torch.no_grad():
    for (x, y) in dataset:
      x = x.to(device)
      prediction = model(x.unsqueeze(0).detach())

      predicted_y = prediction
      predicted_y = predicted_y.squeeze(0).squeeze(0).detach().cpu().numpy()

      x = x.squeeze(0).detach().cpu().numpy()
      y = y.squeeze(0).detach().cpu().numpy()

      all_predicted_ys.append(predicted_y)
      all_xs.append(x)
      all_ys.append(y)
  
  return all_xs, all_ys, all_predicted_ys

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  
  dataset_class = train.get_dataset_class(args)
  dataset = dataset_class('test', polar=args.polar, hospital_id=args.hospital_id, depth=args.depth)

  model = get_model(args, dataset_class, device)
  model.to(device)
  model.load_state_dict(torch.load(args.weights))
  model.eval()
  model.train(False)

  _, all_ys, all_predicted_ys = get_predictions(model, dataset, device)

  if args.polar:
    centers = dataset.centers

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
        y = all_ys[i]
        if args.depth:
          y = y[1]
        results[file_name]['gt'] = (center, y)

  processed_preds = []
  processed_gts = []

  for file_name in results.keys():
    gt_center, gt = results[file_name]['gt']

    for center, pred in results[file_name]['preds']:
      if np.all(center == gt_center):
        break

    for center, curr_pred in results[file_name]['preds']:
      if not np.all(center == gt_center):
        curr_pred = polar_transformations.to_cart(curr_pred, center)
        curr_pred = polar_transformations.to_polar(curr_pred, gt_center)
        pred += curr_pred

    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    processed_preds.append(pred)
    processed_gts.append(gt)

  dscs = np.array([dsc(processed_preds[i], processed_gts[i]) for i in range(len(processed_gts))])
  ious = np.array([iou(processed_preds[i], processed_gts[i]) for i in range(len(processed_gts))])
  precisions = np.array([precision(processed_preds[i], processed_gts[i]) for i in range(len(processed_gts))])
  recalls = np.array([recall(processed_preds[i], processed_gts[i]) for i in range(len(processed_gts))])

  print(f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')
  return dscs.mean(), ious.mean(), precisions.mean(), recalls.mean()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Testing'
  )
  parser.add_argument(
    '--weights', type=str, help='path to weights'
  )
  parser.add_argument(
    '--model', type=str, choices=train.model_choices, default='liver', help='dataset type'
  )
  parser.add_argument(
    '--dataset', type=str, choices=train.dataset_choices, default='liver', help='dataset type'
  )
  parser.add_argument(
      '--polar', 
      action='store_true',
      help='use polar coordinates')

  parser.add_argument(
      '--hospital-id', type=str, choices=['D', 'K', 'R'], default='D', help='which dataset subset (center) to use'
  )
  parser.add_argument(
      '--depth', 
      action='store_true',
      help='use 3 consequtive slices')
  args = parser.parse_args()
  main(args)

