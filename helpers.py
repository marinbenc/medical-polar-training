import os

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pydicom

def dsc(y_pred, y_true):

  y_pred[y_pred > 0.5] = 1
  y_pred[y_pred <= 0.5] = 0

  y_true[y_true > 0.5] = 1
  y_true[y_true <= 0.5] = 0

  if not np.any(y_true):
    return 1 - np.sum(y_pred) * 2.0 / np.sum(1 - y_true)
  else:
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

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


def preprocess_dcm_image(image):
  MIN_BOUND = -200
  MAX_BOUND = -30
  GLOBAL_MEAN = 0.0616 # determined by averageing all thresholded and normalized dicom images

  # threshold
  image[image > MAX_BOUND] = 0
  image[image < MIN_BOUND] = 0

  # normalize
  image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
  image[image>1] = 1.
  image[image<0] = 0.

  # invert
  image = 1 - image

  # zero center
  image -= GLOBAL_MEAN

  return image
