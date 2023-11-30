'''

The provided code is a Python script that defines a data loader for loading and processing image data, typically used in machine learning models, especially in image super-resolution tasks. Here's a breakdown of its key functionalities:

Import Libraries: It imports necessary libraries and modules like collections, os, cv2 (OpenCV), numpy, tensorflow, and some specific TensorFlow sub-modules.

Function Definition (inference_data_loader): The main function, inference_data_loader, is defined to load image data for inference (i.e., making predictions with a trained model). This function takes three parameters:

input_lr_dir: Directory containing low-resolution images.
input_hr_dir: Directory containing high-resolution images, optional.
input_dir_len: Length of the input directory, optional.
Directory Validation: The function first checks if the given directories (input_lr_dir and input_hr_dir) exist. If not, it raises a ValueError.

Image File Listing and Sorting: It lists and sorts the image files (specifically .png files) in the specified directory. The sorting is done alphabetically and numerically to ensure a consistent order.

Preprocess Images: For each image, the following preprocessing steps are performed:

Reading the image using OpenCV.
If down_sp is True (indicating a need to downsample), apply a Gaussian blur and downsample the image by a factor of 4.
Normalize the image data by dividing by 255.0, converting pixel values to a range of 0 to 1.
Data Aggregation: The processed images and their paths are aggregated into two lists: image_lr and image_list_lr.

Data Reorganization: The lists are modified to include some reversed order elements, possibly for augmentation or other processing purposes.

Return Data: It returns a named tuple Data containing the paths to the low-resolution images (paths_LR) and the preprocessed image inputs (inputs).

The script is a typical example of a data loading and preprocessing pipeline in machine learning, where images are read, optionally downsampled, normalized, and then made ready for input to a model.
'''

"""Data loader for loading testing data."""

import collections
import os

import cv2

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.io.gfile as gfile


def inference_data_loader(
    input_lr_dir,
    input_hr_dir=None,
    input_dir_len=-1,
):
  """Inference pipeline data loader."""
  filedir = input_lr_dir
  down_sp = False
  if (input_lr_dir is None) or (not gfile.exists(input_lr_dir)):
    if (input_hr_dir is None) or (not gfile.exists(input_hr_dir)):
      raise ValueError('Input directory not found')
    filedir = input_hr_dir
    down_sp = True

  image_list_lr_temp = gfile.listdir(filedir)
  image_list_lr_temp = [_ for _ in image_list_lr_temp if _.endswith('.png')]
  image_list_lr_temp = sorted(
      image_list_lr_temp
  )  # first sort according to abc, then sort according to 123
  image_list_lr_temp.sort(
      key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
  if input_dir_len > 0:
    image_list_lr_temp = image_list_lr_temp[:input_dir_len]

  image_list_lr = [os.path.join(filedir, _) for _ in image_list_lr_temp]

  # Read in and preprocess the images
  def preprocess_test(name):

    with tf.gfile.Open(name, 'rb') as fid:
      raw_im = np.asarray(bytearray(fid.read()), dtype=np.uint8)
      im = cv2.imdecode(raw_im, cv2.IMREAD_COLOR).astype(np.float32)[:, :, ::-1]

    if down_sp:
      icol_blur = cv2.GaussianBlur(im, (0, 0), sigmaX=1.5)
      im = icol_blur[::4, ::4, ::]
    im = im / 255.0
    return im

  image_lr = [preprocess_test(_) for _ in image_list_lr]
  image_list_lr = image_list_lr[5:0:-1] + image_list_lr
  image_lr = image_lr[5:0:-1] + image_lr

  Data = collections.namedtuple('Data', 'paths_LR, inputs')
  return Data(paths_LR=image_list_lr, inputs=image_lr)