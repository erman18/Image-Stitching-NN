import numpy as np
from sklearn.feature_extraction import image
from scipy.misc import imsave, imread, imresize
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter
import os

scaling_factor = 3
fsub = 33
imgsize = 400
nb_images = 38400 # 96 x 400 (each of 33 x 33 pixels)

input_path = r"input_images\\"
output_path_X = r"output_images_X\\"
output_path_Y = r"output_images_Y\\"

if not os.path.exists(output_path_X):
    os.makedirs(output_path_X)

if not os.path.exists(output_path_Y):
    os.makedirs(output_path_Y)

def loadImages():
    import os
    # Hold the images
    dataX = np.zeros((nb_images, 3, fsub, fsub))
    dataY = np.zeros((nb_images, 3, fsub, fsub))

    for i, file in enumerate(os.listdir(output_path_Y)):
        # Training images are blurred versions ('Y' according to paper)
        y = imread(output_path_Y + file, mode="RGB")
        y = y.transpose((2, 0, 1)).astype('float64') / 255
        dataX[i, :, :, :] = y

        # Non blurred images ('X' according to paper)
        x = imread(output_path_X + file, mode="RGB")
        x = x.transpose((2, 0, 1)).astype('float64') / 255
        dataY[i, :, :, :] = x

        if i % 1000 == 0  : print('%f percent loaded.' % (i * 100/ nb_images))
    return (dataX, dataY)

def loadDenoisingImages():
    import os
    # Hold the images
    dataX = np.zeros((nb_images, 3, fsub-1, fsub-1))
    dataY = np.zeros((nb_images, 3, fsub-1, fsub-1))

    for i, file in enumerate(os.listdir(output_path_Y)):
        # Training images are blurred versions ('Y' according to paper)
        y = imread(output_path_Y + file, mode="RGB")
        y = imresize(y, (fsub-1, fsub-1))
        y = y.transpose((2, 0, 1)).astype('float64') / 255
        dataX[i, :, :, :] = y

        # Non blurred images ('X' according to paper)
        x = imread(output_path_X + file, mode="RGB")
        x = imresize(x, (fsub-1, fsub-1))
        x = x.transpose((2, 0, 1)).astype('float64') / 255
        dataY[i, :, :, :] = x

        if i % 1000 == 0: print('%f percent loaded.' % (i * 100 / nb_images))
    return (dataX, dataY)

def transform_images(directory):
    import os
    import time
    index = 1

    # For each image in input_images directory
    for file in os.listdir(directory):
        img = imread(input_path + file, mode='RGB')

        # Resize to 400 x 400
        img = imresize(img, (imgsize, imgsize))

        # Create patches
        patches = image.extract_patches_2d(img, (fsub, fsub), max_patches=imgsize)

        t1 = time.time()
        # Create 400 'X' and 'Y' sub-images of size 33 x 33 for each patch
        for i in range(imgsize):
            ip = patches[i]
            # Save ground truth image X
            imsave(output_path_X + "%d_%d.png" % (index, i+1), ip)

            # Apply Gaussian Blur to Y
            op = gaussian_filter(ip, sigma=0.5)

            # Subsample by scaling factor 3 to Y
            op = imresize(op, (fsub // scaling_factor, fsub // scaling_factor))

            # Upscale by scaling factor 3 to Y
            op = imresize(op, (fsub, fsub), interp='bicubic')

            # Save Y
            imsave(output_path_Y + "%d_%d.png" % (index, i+1), op)

        print("Finished image %d in time %0.2f seconds. (%s)" % (index, time.time() - t1, file))
        index += 1

    print("Images finished.")

def split_image(img, scaling_factor):
    """
    Splits image in (scale_factor x scale_factor) partitions, therefore allowing smaller images of size
    (height / scale_factor, width_scale_factor) to be loaded into gpu for scaling

    :param img: Image of shape (height, width, channels)
    :return: Sharded image of shape (s*s, height, width, channels) where s is scale_factor
    """
    height, width = img.shape[0], img.shape[1]

    shard_height = height // scaling_factor
    shard_width = width // scaling_factor
    nb_shards = scaling_factor * scaling_factor

    # Holder for image shards
    shards = np.empty((nb_shards, shard_height, shard_width, 3))
    shard_index = 0

    for i in range(0, scaling_factor):
        for j in range(0, scaling_factor):
            shards[shard_index, :, :, :] = img[i*shard_height:(i+1)*shard_height,
                                               j*shard_width: (j+1)*shard_width, :]
            shard_index += 1

    return shards

def merge_images(imgs, scaling_factor):
    """
    Merges the shards of the image into a new image of shape (true_height, true_width, 3)

    :return: Merged image of shape (true_height, true_width, 3)
    """
    height, width = imgs.shape[1], imgs.shape[2]
    true_height, true_width = height * scaling_factor, width * scaling_factor

    # Holder for image
    img = np.empty((true_height, true_width, 3))
    img_index = 0

    for i in range(0, scaling_factor):
        for j in range(0, scaling_factor):
            img[i*height:(i+1)*height, j*width: (j+1)*width, :] = imgs[img_index, :, :, :]
            img_index += 1

    return img

def make_patch_grid(x, scale, patch_size, patch_stride=1, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    #x = x.transpose(2, 1, 0)
    height, width = x.shape[0:2]
    x = imresize(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def combine_patches_grid(in_patches, out_shape, scale):
    '''Reconstruct an image from these `patches`

    input shape: (rows, cols, channels, patch_row, patch_col)
    '''
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon


if __name__ == "__main__":
    # Transform the images once, then run the main code to scale images
    #transform_images(input_path)
    pass