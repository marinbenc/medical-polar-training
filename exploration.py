import os

import sys
sys.path.append('../../..')

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.patches import Rectangle

sys.path.append('datasets/polyp')
from polyp_dataset import PolypDataset
from heatmap_dataset import HeatmapDataset

dataset = HeatmapDataset(PolypDataset('valid', polar=False))

for (volume, mask) in dataset:
  volume = np.array(volume)
  volume = volume.transpose(1, 2, 0) + 0.5
  mask = np.array(mask).squeeze()

  fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
  ax1.imshow(volume)
  ax1.set_title('Image')

  ax2.imshow(mask)
  ax2.set_title('Mask')

  plt.show()


# volume_path = 'train/volume-4.nii'
# segmentation_path = 'train/segmentation-4.nii'

# volume = np.rot90(nib.load(volume_path).get_fdata())
# segmentation = np.rot90(nib.load(segmentation_path).get_fdata())

# mask = segmentation[:, :, 577]
# mask[mask > 1] = 1

# print(volume[..., 0].shape)


# volume_slice = volume[:, :, 577]


# center = polar_transformations.centroid(segmentation[:, :, 577])

# volume_slice[volume_slice > 200] = 200
# volume_slice[volume_slice < 0] = 0
# volume_slice /= 200.0
# volume_slice -= 0.07

# print(volume_slice.min(), volume_slice.max(), volume_slice.mean())

# print(volume_slice.mean())

# volume_slice = polar_transformations.to_polar(volume_slice, center)

# fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
# ax1.imshow(volume_slice)
# ax1.set_title('Image')
# ax2.imshow(polar_transformations.to_polar(mask, center))
# ax2.set_title('Mask')
# plt.show()