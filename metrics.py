'''

This code snippet defines a set of utility functions for evaluating image quality, particularly in the context of image processing or super-resolution models. These functions provide methods for calculating standard image quality metrics and performing necessary pre-processing steps. Here's a breakdown of each function and its purpose:

list_png_in_dir(dirpath): Lists all .png files in a given directory (dirpath). It filters out files starting with 'IB' and sorts the remaining files in a specific numeric order. This is likely used to prepare a dataset for evaluation.

rgb_to_ycbcr(img, max_val): Converts an RGB image to the YCbCr color space. This transformation is common in image processing, particularly in tasks like super-resolution, where working in YCbCr can be more effective.

to_uint8(x, vmin, vmax): Normalizes and scales a floating-point array to the 0-255 range, converting it to an 8-bit unsigned integer format. This is a common step before calculating quality metrics on images.

psnr(img_true, img_pred, y_channel): Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images. If y_channel is True, it converts images to YCbCr and uses only the Y (luminance) channel. PSNR is a widely used metric to measure the quality of reconstruction of lossy compression codecs.

ssim(img_true, img_pred, y_channel): Calculates the Structural Similarity Index Measure (SSIM) between two images. Similar to PSNR, it optionally converts images to YCbCr and computes the metric on the Y channel. SSIM is another common metric for assessing the perceived quality of images.

crop_8x8(img): Crops an image to a size that is a multiple of 32, subtracting up to 16 pixels from each dimension if necessary. This might be used to prepare images for a model that requires input dimensions to be multiples of a certain number.

The overall purpose of these functions is to facilitate the evaluation of image processing algorithms, particularly in the context of tasks like image super-resolution, where comparing the quality of output images against ground truths is crucial. These functions help in preprocessing images, converting them into the correct format, and calculating standard metrics like PSNR and SSIM.
'''

"""Metrics for eval."""

import os

from absl import flags
import numpy as np

from skimage import metrics

import tensorflow.compat.v1.io.gfile as gfile

FLAGS = flags.FLAGS


def list_png_in_dir(dirpath):
  """List all directoties under dirpath."""
  filelist = gfile.listdir(dirpath)
  filelist = [_ for _ in filelist if _.endswith('.png')]
  filelist = [_ for _ in filelist if not _.startswith('IB')]
  filelist = sorted(filelist)
  filelist.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
  result = [os.path.join(dirpath, _) for _ in filelist if _.endswith('.png')]
  return result


def rgb_to_ycbcr(img, max_val=255):
  """color space transform, from https://github.com/yhjo09/VSR-DUF."""
  o = np.array([[16], [128], [128]])
  trans = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                    [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                    [0.439215686274510, -0.367788235294118,
                     -0.071427450980392]])

  if max_val == 1:
    o = o / 255.0

  t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
  t = np.dot(t, np.transpose(trans))
  t[:, 0] += o[0]
  t[:, 1] += o[1]
  t[:, 2] += o[2]
  ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

  return ycbcr


def to_uint8(x, vmin, vmax):
  # color space transform, originally from https://github.com/yhjo09/VSR-DUF
  x = x.astype('float32')
  x = (x - vmin) / (vmax - vmin) * 255  # 0~255
  return np.clip(np.round(x), 0, 255)


def psnr(img_true, img_pred, y_channel=True):
  """PSNR with color space transform, originally."""
  if y_channel:
    y_true = rgb_to_ycbcr(to_uint8(img_true, 0, 255), 255)[:, :, 0]
    y_pred = rgb_to_ycbcr(to_uint8(img_pred, 0, 255), 255)[:, :, 0]
  else:
    y_true = to_uint8(img_true, 0, 255)
    y_pred = to_uint8(img_pred, 0, 255)
  diff = y_true - y_pred
  rmse = np.sqrt(np.mean(np.power(diff, 2)))
  return 20 * np.log10(255. / rmse)


def ssim(img_true, img_pred, y_channel=True):
  """SSIM with color space transform."""
  if y_channel:
    y_true = rgb_to_ycbcr(to_uint8(img_true, 0, 255), 255)[:, :, 0]
    y_pred = rgb_to_ycbcr(to_uint8(img_pred, 0, 255), 255)[:, :, 0]
  else:
    y_true = to_uint8(img_true, 0, 255)
    y_pred = to_uint8(img_pred, 0, 255)

  ssim_val = metrics.structural_similarity(
      y_true,
      y_pred,
      data_range=y_pred.max() - y_pred.min(),
      multichannel=not y_channel)
  return ssim_val


def crop_8x8(img):
  """Crop 8x8 of the input image."""
  ori_h = img.shape[0]
  ori_w = img.shape[1]

  h = (ori_h // 32) * 32
  w = (ori_w // 32) * 32

  while h > ori_h - 16:
    h = h - 32
  while w > ori_w - 16:
    w = w - 32

  y = (ori_h - h) // 2
  x = (ori_w - w) // 2
  crop_img = img[y:y + h, x:x + w]
  return crop_img, y, x