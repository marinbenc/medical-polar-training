import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append('models')
from unet_plain import UNet

sys.path.append('datasets/liver')
from liver_dataset import LiverDataset
sys.path.append('datasets/polyp')
from polyp_dataset import PolypDataset

from helpers import dsc

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

      all_predicted_ys.append(predicted_y)

      x = x.squeeze(0).detach().cpu().numpy()
      all_xs.append(x)

      y = y.squeeze(0).detach().cpu().numpy()
      all_ys.append(y)

      # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
      # ax1.imshow(y)
      # ax1.set_title('GT')
      # ax2.imshow(predicted_y.squeeze())
      # ax2.set_title('Predicted')
      # plt.show()
  
  return all_xs, all_ys, all_predicted_ys

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

  if args.dataset == 'liver':
    dataset_class = LiverDataset
  elif args.dataset == 'polyp':
    dataset_class = PolypDataset
    
  dataset = dataset_class('test', polar=args.polar)

  model = UNet(
    in_channels=dataset_class.in_channels, 
    out_channels=dataset_class.out_channels, 
    device=device)
  model.to(device)
  model.load_state_dict(torch.load(args.weights))
  model.eval()
  model.train(False)

  _, all_ys, all_predicted_ys = get_predictions(model, dataset, device)

  dscs = np.array([dsc(all_ys[i], all_predicted_ys[i]) for i in range(len(all_ys))])
  print(f'Mean DSC: {dscs.mean()}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Testing'
  )
  parser.add_argument(
    '--weights', type=str, help='path to weights'
  )
  parser.add_argument(
    '--dataset', type=str, choices=['liver', 'polyp'], default='liver', help='dataset type'
  )
  parser.add_argument(
      '--polar', 
      action='store_true',
      help='use polar coordinates')

  args = parser.parse_args()
  main(args)

