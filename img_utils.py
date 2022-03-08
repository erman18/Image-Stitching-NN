from __future__ import print_function, division, absolute_import

import numpy as np
# from scipy.misc import imsave, imread, imresize
from imageio import imwrite, imread
from cv2 import resize, INTER_CUBIC
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter

import patchify

from keras import backend as K

import os
import time

import tensorflow as tf
import numpy as np
import PIL.Image
import os
import json
import prepare_data as psd
import constant as cfg

# from hparams import hparams
hparams = {'input_size': (512, 512, 3),
           'batch_size': 4,
           'content_weight': 1e-3, # 1e-5,
           'style_weight': 4e-9, # 4e-9,
           'simple_weight': 2e-3, # 4e-5, # 4e-9,
           'gradient_weight': 4e-1,
           'learning_rate': 0.001,
           'residual_filters': 128,
           'residual_layers': 5,
           'initializer': 'glorot_normal',
           'style_layers': ['block1_conv2',
                            'block2_conv2',
                            'block3_conv3',
                            'block4_conv3'],
           'content_layer_index': 2,
           'test_size': (512, 512, 3)
}


def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

def convert(file_path, shape):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, shape)
    return img

def tensor_to_image(tensor):
    tensor = 255*(tensor + 1.0)/2.0
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def gram_matrix(input_tensor):
    input_tensor = tf.cast(input_tensor, tf.float32) # avoid mixed_precision nan
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32) # int32 to float32
    return result/num_locations

def content_loss(content, output):
    return tf.reduce_mean((content-output)**2)

def gradient_loss(content, output):
    "Ensure the smoothness of the images"

    # kernels = np.ones((9,9,3,1), dtype=np.float32) / 81.0 # * np.array([0.2126, 0.7152, 0.0722])
    # kernels_tf = tf.constant(kernels, dtype=tf.float32)

    # # Output tensor has shape [batch_size, h, w, d * num_kernels].
    # strides = [1, 1, 1, 1]

    dy_content, dx_content = tf.image.image_gradients(content)
    dy_ouput, dx_ouput = tf.image.image_gradients(output)

    ly = (dy_content - dy_ouput)**2
    lx = (dx_content - dx_ouput)**2

    # lx_f = tf.nn.conv2d(lx, kernels_tf, strides=strides, padding='SAME')
    # ly_f = tf.nn.conv2d(ly, kernels_tf, strides=strides, padding='SAME')

    return tf.reduce_mean(lx + ly)

def style_loss(style, output):
    return tf.add_n([tf.reduce_mean((style_feat-out_feat)**2) 
                        for style_feat, out_feat in zip(style, output)])

def save_hparams(model_name):
    json_hparams = json.dumps(hparams)
    f = open(os.path.join(cfg.project_dir, '{}_hparams.json'.format(model_name)), 'w')
    f.write(json_hparams)
    f.close()

def save_dataset_indexes(model_name, train_indexes, test_indexes, samples_per_epoch, val_count):
    hindexses = {'train_indexes': train_indexes,
                'test_indexes': test_indexes,
                'samples_per_epoch': samples_per_epoch,
                'val_count': val_count}
    filename = os.path.join(cfg.project_dir, '{}_index.json'.format(model_name))
    cfg.PRINT_INFO(f"Saving dataset indexes: {filename}")
    psd.write_json_file(filename, hindexses)

def get_dataset_indexes(model_name):
    filename = os.path.join(cfg.project_dir, '{}_index.json'.format(model_name))
    cfg.PRINT_INFO(f"Reading dataset indexes: {filename} ...")
    config_data = psd.read_json_file(filename)
    return config_data

def delete_dataset_indexes(model_name):
    filename = os.path.join(cfg.project_dir, '{}_index.json'.format(model_name))
    cfg.PRINT_INFO(f"Delete dataset indexes: {filename} ...")
    config_data = psd.remove_file(filename)

def make_patches(x, scale, patch_size, upscale=True, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    height, width = x.shape[:2]
    if upscale: x = resize(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def patchyfy(img, patch_shape, step=32):
    X, Y, c = img.shape
    x, y = patch_shape
    shape = (X - x + step, Y - y + step, x, y, c)
    X_str, Y_str, c_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str, c_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def make_raw_patches(x, patch_size, channels=3, step=1, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    patches = patchify.patchify(x, (patch_size, patch_size, channels), step=step)
    return patches


# def combine_patches(in_patches, out_shape, scale):
#     '''Reconstruct an image from these `patches`'''
#     recon = reconstruct_from_patches_2d(in_patches, out_shape)
#     return recon


# def image_generator(directory, scale_factor=2, target_shape=None, channels=3, small_train_images=False, shuffle=True,
#                     batch_size=32, nb_inputs=1, seed=None):
#     if not target_shape:
#         if small_train_images:
#             # if K.image_dim_ordering() == "th":
#             #     image_shape = (channels, 16 * _image_scale_multiplier, 16 * _image_scale_multiplier)
#             #     y_image_shape = (channels, 16 * scale_factor * _image_scale_multiplier,
#             #                      16 * scale_factor * _image_scale_multiplier)
#             # else:
#             #     # image_shape = (16 * _image_scale_multiplier, 16 * _image_scale_multiplier, channels)
#             #     # y_image_shape = (16 * scale_factor * _image_scale_multiplier,
#             #     #                  16 * scale_factor * _image_scale_multiplier, channels)
#             #     image_shape = (32 * _image_scale_multiplier, 32 * _image_scale_multiplier, channels)
#             #     y_image_shape = (32 * scale_factor * _image_scale_multiplier,
#             #                      32 * scale_factor * _image_scale_multiplier, channels)
#             image_shape = (32 * _image_scale_multiplier, 32 * _image_scale_multiplier, channels)
#             y_image_shape = (32 * scale_factor * _image_scale_multiplier,
#                              32 * scale_factor * _image_scale_multiplier, channels)
#         else:
#             # if K.image_dim_ordering() == "th":
#             #     image_shape = (
#             #     channels, 32 * scale_factor * _image_scale_multiplier, 32 * scale_factor * _image_scale_multiplier)
#             #     y_image_shape = image_shape
#             # else:
#             #     image_shape = (32 * scale_factor * _image_scale_multiplier, 32 * scale_factor * _image_scale_multiplier,
#             #                    channels)
#             #     y_image_shape = image_shape
#             image_shape = (32 * scale_factor * _image_scale_multiplier, 32 * scale_factor * _image_scale_multiplier,
#                            channels)
#             y_image_shape = image_shape
#     else:
#         if small_train_images:
#             # if K.image_dim_ordering() == "th":
#             #     y_image_shape = (3,) + target_shape
#             #
#             #     target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
#             #                     target_shape[1] * _image_scale_multiplier // scale_factor)
#             #     image_shape = (3,) + target_shape
#             # else:
#             #     y_image_shape = target_shape + (channels,)
#             #
#             #     target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
#             #                     target_shape[1] * _image_scale_multiplier // scale_factor)
#             #     image_shape = target_shape + (channels,)
#             y_image_shape = target_shape + (channels,)

#             target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
#                             target_shape[1] * _image_scale_multiplier // scale_factor)
#             image_shape = target_shape + (channels,)
#         else:
#             # if K.image_dim_ordering() == "th":
#             #     image_shape = (channels,) + target_shape
#             #     y_image_shape = image_shape
#             # else:
#             #     image_shape = target_shape + (channels,)
#             #     y_image_shape = image_shape
#             image_shape = target_shape + (channels,)
#             y_image_shape = image_shape

#     file_names = [f for f in sorted(os.listdir(directory + "X/"))]
#     X_filenames = [os.path.join(directory, "X", f) for f in file_names]
#     y_filenames = [os.path.join(directory, "y", f) for f in file_names]

#     nb_images = len(file_names)
#     print("Found %d images." % nb_images)

#     index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

#     while 1:
#         index_array, current_index, current_batch_size = next(index_generator)
#         # print("-----------image_shape: ", image_shape, "- current_batch_size", current_batch_size,
#         #       "- y_image_shape", y_image_shape, "- _image_scale_multiplier: ", _image_scale_multiplier,
#         #       "- small_train_images: ", small_train_images)

#         batch_x = np.zeros((current_batch_size,) + image_shape)
#         batch_y = np.zeros((current_batch_size,) + y_image_shape)

#         for i, j in enumerate(index_array):
#             x_fn = X_filenames[j]
#             img = imread(x_fn, pilmode='RGB')

#             if small_train_images:
#                 img = resize(img, (32 * _image_scale_multiplier, 32 * _image_scale_multiplier))
#             else:
#                 img = resize(img, (image_shape[0], image_shape[1]))
#             img = img.astype('float32') / 255.
#             # print(patchIdx, "---- img_shape: ", img.shape, "- batch_x.shape: ", batch_x[patchIdx].shape,
#             #       "- batch_y.shape: ", batch_y[patchIdx].shape, "- x filename: ", x_fn, "- y filename: ", y_filenames[j])

#             # if K.image_dim_ordering() == "th":
#             #     batch_x[patchIdx] = img.transpose((2, 0, 1))
#             # else:
#             #     batch_x[patchIdx] = img
#             batch_x[i] = img

#             y_fn = y_filenames[j]
#             img = imread(y_fn, pilmode="RGB")
#             img = img.astype('float32') / 255.

#             # if K.image_dim_ordering() == "th":
#             #     batch_y[patchIdx] = img.transpose((2, 0, 1))
#             # else:
#             #     batch_y[patchIdx] = img
#             batch_y[i] = img

#         if nb_inputs == 1:
#             yield batch_x, batch_y
#         else:
#             batch_x = [batch_x for i in range(nb_inputs)]
#             yield batch_x, batch_y


# def image_stitching_generator(directory, scale_factor=2, target_shape=None, channels=3, small_train_images=False, shuffle=True,
#                     batch_size=32, nb_inputs=1, seed=None):

#     if not target_shape:
#         if small_train_images:
#             image_shape = (32 * _image_scale_multiplier, 32 * _image_scale_multiplier, channels)
#             y_image_shape = (32 * scale_factor * _image_scale_multiplier,
#                              32 * scale_factor * _image_scale_multiplier, channels)
#         else:
#             image_shape = (32 * scale_factor * _image_scale_multiplier, 32 * scale_factor * _image_scale_multiplier,
#                            channels)
#             y_image_shape = image_shape
#     else:
#         if small_train_images:
#             y_image_shape = target_shape + (channels,)

#             target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
#                             target_shape[1] * _image_scale_multiplier // scale_factor)
#             image_shape = target_shape + (channels,)
#         else:
#             image_shape = target_shape + (channels,)
#             y_image_shape = image_shape

#     file_names = [f for f in sorted(os.listdir(directory + "X/"))]
#     X_filenames = [os.path.join(directory, "X", f) for f in file_names]
#     y_filenames = [os.path.join(directory, "y", f) for f in file_names]

#     nb_images = len(file_names)
#     print("Found %d images." % nb_images)

#     index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

#     while 1:
#         index_array, current_index, current_batch_size = next(index_generator)
#         # print("-----------image_shape: ", image_shape, "- current_batch_size", current_batch_size,
#         #       "- y_image_shape", y_image_shape, "- _image_scale_multiplier: ", _image_scale_multiplier,
#         #       "- small_train_images: ", small_train_images)

#         batch_x = np.zeros((current_batch_size,) + image_shape)
#         batch_y = np.zeros((current_batch_size,) + y_image_shape)

#         for i, j in enumerate(index_array):
#             x_fn = X_filenames[j]
#             img = imread(x_fn, pilmode='RGB')

#             if small_train_images:
#                 img = resize(img, (32 * _image_scale_multiplier, 32 * _image_scale_multiplier))
#             else:
#                 img = resize(img, (image_shape[0], image_shape[1]))
#             img = img.astype('float32') / 255.
            
#             batch_x[i] = img

#             y_fn = y_filenames[j]
#             img = imread(y_fn, pilmode="RGB")
#             img = img.astype('float32') / 255.

#             batch_y[i] = img

#         if nb_inputs == 1:
#             yield batch_x, batch_y
#         else:
#             batch_x = [batch_x for i in range(nb_inputs)]
#             yield batch_x, batch_y


# def _index_generator(N, batch_size=32, shuffle=True, seed=None):
#     batch_index = 0
#     total_batches_seen = 0

#     while 1:
#         if seed is not None:
#             np.random.seed(seed + total_batches_seen)

#         if batch_index == 0:
#             index_array = np.arange(N)
#             if shuffle:
#                 index_array = np.random.permutation(N)

#         current_index = (batch_index * batch_size) % N

#         if N >= current_index + batch_size:
#             current_batch_size = batch_size
#             batch_index += 1
#         else:
#             current_batch_size = N - current_index
#             batch_index = 0
#         total_batches_seen += 1

#         yield (index_array[current_index: current_index + current_batch_size],
#                current_index, current_batch_size)


# def smooth_gan_labels(y):
#     assert len(y.shape) == 2, "Needs to be a binary class"
#     y = np.asarray(y, dtype='int')
#     Y = np.zeros(y.shape, dtype='float32')

#     for i in range(y.shape[0]):
#         for j in range(y.shape[1]):
#             if y[i, j] == 0:
#                 Y[i, j] = np.random.uniform(0.0, 0.3)
#             else:
#                 Y[i, j] = np.random.uniform(0.7, 1.2)

#     return Y


if __name__ == "__main__":
    # Transform the images once, then run the main code to scale images

    # Change scaling factor to increase the scaling factor
    scaling_factor = 2

    # Set true_upscale to True to generate smaller training images that will then be true upscaled.
    # Leave as false to create same size input and output images
    true_upscale = True

    # transform_images_temp(input_path, output_path, scaling_factor=scaling_factor, max_nb_images=-1,
    #                  true_upscale=true_upscale)
    # transform_images_temp(validation_set5_path, validation_output_path, scaling_factor=scaling_factor, max_nb_images=-1,
    #                       true_upscale=true_upscale)
    # transform_images_temp(validation_set14_path, validation_output_path, scaling_factor=scaling_factor, max_nb_images=-1,
    #                       true_upscale=true_upscale)
    pass
