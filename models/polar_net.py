from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import polar_transformations
import numpy as np
import cv2 as cv

import sys
sys.path.append('models/stacked_hourglass')
from stacked_hourglass import StackedHourglass

class PolarNet(nn.Module):

  counter = 0

  def __init__(self, center_model, seg_model):
    super(PolarNet, self).__init__()

    self.center_model = center_model
    self.seg_model = seg_model

  def polar_transform():
    i = x - center.x

  def forward(self, x):
    x_center, x_seg = x

    test = torch.zeros_like(x_seg)
    for i in range(x_seg.shape[0]):
      circle = np.zeros((test.shape[-2], test.shape[-1], 3))
      print(circle.shape)
      circle = cv.circle(circle, (circle.shape[1] // 2, circle.shape[0] // 2), circle.shape[0] // 3, (255, 0, 0), 10)
      circle = circle.astype(np.float32) / 255.0
      test[i] = torch.from_numpy(circle.transpose(2, 0, 1)).unsqueeze(0)
      plt.imshow(circle)
      plt.show()
    
    x_seg = test

    heatmaps = self.center_model(x_center)
    last_heatmaps = heatmaps[:, -1, :, :]

    n = last_heatmaps.shape[0]
    d = last_heatmaps.shape[-1]

    centers = last_heatmaps.view(n, -1).argmax(1).view(-1, 1).float()
    centers = torch.cat((centers // d, centers % d), dim=1)

    centers[:, 0] = 32
    centers[:, 1] = 32

    centers[:, 0] /= last_heatmaps[0].shape[-1]
    centers[:, 1] /= last_heatmaps[0].shape[-2]

    centers[:, 0] *= x_seg[0].shape[-1]
    centers[:, 1] *= x_seg[0].shape[-2]


    original_image_dimensions = x_seg.shape[-2:]
    output_shape = (x_seg.shape[0], 1, *original_image_dimensions)
    transposed_image_shape = (*x_seg.shape[:-2], *original_image_dimensions[::-1])

    x_seg_polar = torch.zeros(transposed_image_shape, device=x_seg.device)

    for i in range(x_seg.shape[0]):
      x_np = x_seg[i].detach().cpu().numpy().squeeze()
      x_np = x_np.transpose(1, 2, 0)
      center_np = centers[i].detach().cpu().numpy()[::-1]
      x_polar = polar_transformations.to_polar(x_np, tuple(center_np))
      # plt.imshow(x_polar)
      # plt.show()
      x_polar = x_polar.transpose(2, 0, 1)
      x_seg_polar[i] = torch.from_numpy(x_polar)

    x_seg = x_seg_polar[:, 0, :, :]#self.seg_model(x_seg_polar)
    
    x_seg_cart = torch.zeros(output_shape, device=x_seg.device)

    # test = torch.zeros_like(x_seg)
    # for i in range(x_seg.shape[0]):
    #   circle = np.zeros((test.shape[-2], test.shape[-1]))
    #   print(circle.shape)
    #   circle = cv.line(circle, (0, circle.shape[0] // 2), (circle.shape[1], circle.shape[0] // 2), 255, 10)
    #   circle = circle.astype(np.float32) / 255.0
    #   test[i] = torch.from_numpy(circle).unsqueeze(0)
    
    # x_seg = test
      


    for i in range(x_seg.shape[0]):
      x_np = x_seg[i].detach().cpu().numpy().squeeze()
      # plt.imshow(x_np)
      # plt.show()
      center_np = centers[i].detach().cpu().numpy()[::-1]
      x_cart = polar_transformations.to_cart(x_np * 255, tuple(center_np))
      x_cart = x_cart.astype(np.float32) / 255.0
      plt.imshow(x_cart)
      plt.show()
      x_seg_cart[i] = torch.from_numpy(x_cart).unsqueeze(0)

    # PolarNet.counter += 1
    # if PolarNet.counter > 500:
    #   plt.imshow(x_seg_cart[0].detach().cpu().numpy().squeeze())
    #   plt.show()

    return (heatmaps, x_seg_cart)
