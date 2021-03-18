import os

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def centroid(img):
  if len(img.shape) > 2:
    M = cv.moments(img[:,:,1])
  else:
    M = cv.moments(img)

  if M["m00"] == 0:
    return (img.shape[0] // 2, img.shape[1] // 2)
  
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  return (cX, cY)

def to_polar(input_img, center):
  input_img = input_img.astype(np.float32)
  value = np.sqrt(((input_img.shape[0]/2.0)**2.0)+((input_img.shape[1]/2.0)**2.0))
  polar_image = cv.linearPolar(input_img, center, value, cv.WARP_FILL_OUTLIERS)
  polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
  return polar_image

def to_cart(input_img, center):
  input_img = input_img.astype(np.float32)
  input_img = cv.rotate(input_img, cv.ROTATE_90_CLOCKWISE)
  value = np.sqrt(((input_img.shape[1]/2.0)**2.0)+((input_img.shape[0]/2.0)**2.0))
  polar_image = cv.linearPolar(input_img, center, value, cv.WARP_FILL_OUTLIERS + cv.WARP_INVERSE_MAP)
  polar_image = polar_image.astype(np.uint8)
  return polar_image

if __name__ == "__main__":
  image = cv.imread('test_images/30.tif')
  plt.imshow(image)
  plt.show()

  center = centroid(image)
  
  polar = to_polar(image, center)
  plt.imshow(polar)
  plt.show()
  
  cart = to_cart(polar, center)
  plt.imshow(cart)
  plt.show()
