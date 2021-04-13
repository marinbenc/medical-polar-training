import os

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from skimage.metrics import adapted_rand_error

def _thresh(img):
  img[img > 0.5] = 1
  img[img <= 0.5] = 0
  return img

def dsc(y_pred, y_true):
  y_pred = _thresh(y_pred)
  y_true = _thresh(y_true)

  if not np.any(y_true):
    return 1 - np.sum(y_pred) * 2.0 / np.sum(1 - y_true)
  else:
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def iou(y_pred, y_true):
  y_pred = _thresh(y_pred)
  y_true = _thresh(y_true)

  intersection = np.logical_and(y_pred, y_true)
  union = np.logical_or(y_pred, y_true)
  return intersection.sum() / float(union.sum())

def precision(y_pred, y_true):
  y_pred = _thresh(y_pred).astype(np.int)
  y_true = _thresh(y_true).astype(np.int)

  if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, precision is 1
    # otherwise it's 0
    return 1 if y_pred.sum() <= 5 else 0

  if y_pred.sum() <= 5:
    return 0.

  _, precision, _ = adapted_rand_error(y_true, y_pred)
  return precision

def recall(y_pred, y_true):
  y_pred = _thresh(y_pred).astype(np.int)
  y_true = _thresh(y_true).astype(np.int)

  if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, recall is 1
    # otherwise it's 0
    return 1 if y_pred.sum() <= 5 else 0

  if y_pred.sum() <= 5:
    return 0.

  _, _, recall = adapted_rand_error(y_true, y_pred)
  return recall

def listdir(path):
  """ List files but remove hidden files from list """
  return [item for item in os.listdir(path) if item[0] != '.']

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def show_images_row(imgs, titles=None, rows=1, figsize=(6.4, 4.8), **kwargs):
  '''
      Display grid of cv2 images
      :param img: list [cv::mat]
      :param title: titles
      :return: None
  '''
  assert ((titles is None) or (len(imgs) == len(titles)))
  num_images = len(imgs)

  if titles is None:
      titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

  fig = plt.figure(figsize=figsize)
  for n, (image, title) in enumerate(zip(imgs, titles)):
      ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
      plt.imshow(image, **kwargs)
      ax.set_title(title)
      plt.axis('off')
  plt.show()
