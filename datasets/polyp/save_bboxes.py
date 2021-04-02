import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.patches import Rectangle

sys.path.append('../..')
import helpers as h

def get_bounding_box(mask):
  # https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/utils.py
  horizontal_indices = np.where(np.any(mask, axis=0))[0]
  vertical_indices = np.where(np.any(mask, axis=1))[0]
  if horizontal_indices.shape[0]:
    x1, x2 = horizontal_indices[[0, -1]]
    y1, y2 = vertical_indices[[0, -1]]

    x1 -= 1
    y1 -= 1
    # x2 and y2 should not be part of the box. Increment by 1.
    x2 += 1
    y2 += 1
  else:
    # No mask for this instance. Might happen due to
    # resizing or cropping. Set bbox to zeros
    x1, x2, y1, y2 = 0, 0, 0, 0

  # normalize coords
  height, width = mask.shape[-2:]
  x1 = x1 / width
  x2 = x2 / width
  y1 = y1 / height
  y2 = y2 / height

  return np.array([x1, y1, x2, y2])

folders = ['train', 'test', 'valid']
output_folders = ['train_bboxes', 'test_bboxes', 'valid_bboxes']

for folder in folders:
  files = h.listdir(os.path.join(folder, 'label'))
  files.sort()

  for file in files:
    mask = cv.imread(os.path.join(folder, 'label', file), cv.IMREAD_GRAYSCALE)
    bbox = get_bounding_box(mask)
    file_name = file.replace('.png', '.csv')


