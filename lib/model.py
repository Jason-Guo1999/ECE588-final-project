'''
The provided code is a part of a machine learning model architecture, specifically designed for image super-resolution tasks. This code appears to be related to the implementation of a model described in the TecoGAN paper, which is a type of Generative Adversarial Network (GAN) used for enhancing the resolution of images. Here's a breakdown of the functions and their roles:

Import Libraries: It imports TensorFlow (both version 1 and 2) and some custom operations from a module named ops.

Flow Network (fnet): The fnet function defines a flow network, which is used for predicting motion in images. It consists of downsampling (encoder) and upsampling (decoder) blocks. The network applies convolutional layers with LeakyReLU activation and max pooling for downsampling, and uses transposed convolutions for upsampling.

Generator Encoder (generator_f_encoder): This function builds the encoder part of the generator. It uses several residual blocks, each consisting of convolutional layers followed by ReLU activation. The residual blocks help in preserving the image information over deep networks.

Generator Decoder (generator_f_decoder): This function builds the decoder part of the generator. It upsamples the encoded features to the higher resolution and then adds the bicubically upscaled low-resolution input to it. This is a common technique in super-resolution models, helping to add fine details to the upscaled image.

Generator Function (generator_f): This function combines the encoder and decoder functions to form the complete generator model of the GAN. It takes low-resolution inputs and generates high-resolution outputs.

Each of these components plays a crucial role in the image super-resolution process:

The flow network (fnet) likely predicts how pixels move between frames in a video sequence, which is useful for temporal consistency in video super-resolution.
The generator uses an encoder-decoder architecture, common in GANs, where the encoder compresses the input and the decoder reconstructs the high-resolution image from this compressed representation.
The residual blocks in the encoder help in retaining the original content while adding new details in the upscaling process.
Overall, this code represents a sophisticated deep learning architecture for enhancing the resolution and quality of images, particularly useful in video applications.
'''
"""Model functions to reconstruct models."""

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from lib import ops


# Definition of the fnet, more details can be found in TecoGAN paper
def fnet(fnet_input, reuse=False):
  """Flow net."""
  def down_block(inputs, output_channel=64, stride=1, scope='down_block'):
    with tf.variable_scope(scope):
      net = ops.conv2(
          inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
      net = ops.lrelu(net, 0.2)
      net = ops.conv2(
          net, 3, output_channel, stride, use_bias=True, scope='conv_2')
      net = ops.lrelu(net, 0.2)
      net = ops.maxpool(net)

    return net

  def up_block(inputs, output_channel=64, stride=1, scope='up_block'):
    with tf.variable_scope(scope):
      net = ops.conv2(
          inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
      net = ops.lrelu(net, 0.2)
      net = ops.conv2(
          net, 3, output_channel, stride, use_bias=True, scope='conv_2')
      net = ops.lrelu(net, 0.2)
      new_shape = tf.shape(net)[1:-1] * 2
      net = tf2.image.resize(net, new_shape)

    return net

  with tf.variable_scope('autoencode_unit', reuse=reuse):
    net = down_block(fnet_input, 32, scope='encoder_1')
    net = down_block(net, 64, scope='encoder_2')
    net = down_block(net, 128, scope='encoder_3')

    net = up_block(net, 256, scope='decoder_1')
    net = up_block(net, 128, scope='decoder_2')
    net1 = up_block(net, 64, scope='decoder_3')

    with tf.variable_scope('output_stage'):
      net = ops.conv2(net1, 3, 32, 1, scope='conv1')
      net = ops.lrelu(net, 0.2)
      net2 = ops.conv2(net, 3, 2, 1, scope='conv2')
      net = tf.tanh(net2) * 24.0
      # the 24.0 is the max Velocity, details can be found in TecoGAN paper
  return net


def generator_f_encoder(gen_inputs, num_resblock=10, reuse=False):
  """Generator function encoder."""
  # The Bx residual blocks
  def residual_block(inputs, output_channel=64, stride=1, scope='res_block'):
    with tf.variable_scope(scope):
      net = ops.conv2(
          inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
      net = tf.nn.relu(net)
      net = ops.conv2(
          net, 3, output_channel, stride, use_bias=True, scope='conv_2')
      net = net + inputs

    return net

  with tf.variable_scope('generator_unit', reuse=reuse):
    # The input layer
    with tf.variable_scope('input_stage'):
      net = ops.conv2(gen_inputs, 3, 64, 1, scope='conv')
      stage1_output = tf.nn.relu(net)

    net = stage1_output

    # The residual block parts
    for i in range(1, num_resblock + 1,
                   1):  # should be 16 for TecoGAN, and 10 for TecoGANmini
      name_scope = 'resblock_%d' % (i)
      net = residual_block(net, 64, 1, name_scope)

  return net


def generator_f_decoder(net,
                        gen_inputs,
                        gen_output_channels,
                        vsr_scale,
                        reuse=False):
  """Generator function decoder."""
  with tf.variable_scope('generator_unit', reuse=reuse):
    with tf.variable_scope('conv_tran2highres'):
      if vsr_scale == 2:
        net = ops.conv2_tran(
            net, kernel=3, output_channel=64, stride=2, scope='conv_tran1')
        net = tf.nn.relu(net)
      if vsr_scale == 4:
        net = ops.conv2_tran(net, 3, 64, 2, scope='conv_tran1')
        net = tf.nn.relu(net)
        net = ops.conv2_tran(net, 3, 64, 2, scope='conv_tran2')
        net = tf.nn.relu(net)

    with tf.variable_scope('output_stage'):
      net = ops.conv2(net, 3, gen_output_channels, 1, scope='conv')
      low_res_in = gen_inputs[:, :, :, 0:3]  # ignore warped pre high res
      bicubic_hi = ops.bicubic_x(low_res_in, scale=vsr_scale)  # can put on GPU
      net = net + bicubic_hi
      net = ops.preprocess(net)
    return net


# Definition of the generator.
def generator_f(gen_inputs,
                gen_output_channels,
                num_resblock=10,
                vsr_scale=4,
                reuse=False):
  net = generator_f_encoder(gen_inputs, num_resblock, reuse)
  net = generator_f_decoder(net, gen_inputs, gen_output_channels, vsr_scale,
                            reuse)

  return net