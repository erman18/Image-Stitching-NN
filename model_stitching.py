from __future__ import print_function, division

from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from advanced import HistoryCheckpoint, SubPixelUpscaling, non_local_block, TensorBoardBatch
import img_utils
from data_generator import DataGenerator, image_stitching_generator
import prepare_stitching_data as psd
from sklearn.model_selection import train_test_split

import numpy as np
import os
import glob
import time
import warnings

try:
    import cv2

    _cv2_available = True
except:
    warnings.warn('Could not load opencv properly. This may affect the quality of output images.')
    _cv2_available = False

train_path = img_utils.output_path
validation_path = img_utils.validation_output_path
path_X = img_utils.output_path + "X/"
path_Y = img_utils.output_path + "y/"


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                    str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


class BaseSuperStitchingModel(object):

    def __init__(self, model_name):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.shape = (None, None, None)
        self.model = None  # type: Model
        self.model_name = model_name
        # self.scale_factor = 1.0
        self.weight_path = None

        # self.type_scale_type = "norm"  # Default = "norm" = 1. / 255
        # self.type_requires_divisible_shape = False
        # self.type_true_upscaling = False

        self.evaluation_func = None
        # self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, nb_camera=5, load_weights=False) -> Model:
        """
        Subclass dependent implementation.
        """
        if width is not None and height is not None:
            self.shape = (width, height, channels * nb_camera)

        init = Input(shape=self.shape)

        return init

    def fit(self, batch_size=32, nb_epochs=100, save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        """

        if self.model is None: self.create_model()

        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=2)]

        # Parameters
        params = {'dim': (self.shape[0], self.shape[1]),
                  'batch_size': batch_size,
                  'n_channels': self.shape[2],
                  'shuffle': True}

        # Datasets
        config_data = psd.read_json_file(psd.config_file)
        data_indexes = np.arange(config_data["total_samples"])
        train_indexes, test_indexes = train_test_split(data_indexes, test_size=0.10)
        samples_per_epoch = len(train_indexes)
        val_count = len(test_indexes)

        # Generators
        training_generator = image_stitching_generator(train_indexes, config_data,
                                                       callee="training_generator", **params)
        validation_generator = image_stitching_generator(test_indexes, config_data,
                                                         callee="validation_generator", **params)
        # training_generator = DataGenerator(train_indexes, config_data, callee="training_generator", **params)
        # validation_generator = DataGenerator(test_indexes, config_data, callee="validation_generator", **params)

        if save_history:
            callback_list.append(HistoryCheckpoint(history_fn))

            if K.backend() == 'tensorflow':
                log_dir = './%s_logs/' % self.model_name
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % self.__class__.__name__)
        # self.model.fit_generator(img_utils.image_stitching_generator(train_path, scale_factor=self.scale_factor,
        #                                                              small_train_images=self.type_true_upscaling,
        #                                                              batch_size=batch_size),
        #                          steps_per_epoch=samples_per_epoch // batch_size + 1,
        #                          epochs=nb_epochs, callbacks=callback_list,
        #                          validation_data=img_utils.image_generator(validation_path,
        #                                                                    scale_factor=self.scale_factor,
        #                                                                    small_train_images=self.type_true_upscaling,
        #                                                                    batch_size=batch_size),
        #                          validation_steps=val_count // batch_size + 1,
        #                          use_multiprocessing=True)

        # Train model on dataset
        # self.model.fit_generator(generator=training_generator,
        #                          steps_per_epoch=samples_per_epoch // batch_size + 1,
        #                          epochs=nb_epochs,
        #                          callbacks=callback_list,
        #                          validation_data=validation_generator,
        #                          validation_steps=val_count // batch_size + 1,
        #                          use_multiprocessing=True,
        #                          workers=2)
        self.model.fit_generator(generator=training_generator,
                                 steps_per_epoch=samples_per_epoch // batch_size + 1,
                                 epochs=nb_epochs,
                                 callbacks=callback_list,
                                 validation_data=validation_generator,
                                 validation_steps=val_count // batch_size + 1,
                                 validation_freq=2,
                                 use_multiprocessing=False)

        return self.model

    def evaluate(self, validation_dir):
        pass

    def stitch(self, img_path, save_intermediate=False, return_image=False, suffix="stitch",
               patch_size=8, mode="patch", verbose=True):
        """
        Standard method to upscale an image.
        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "stitch" + path[1]

        # Read image
        # true_img = imread(img_path)
        # h, w = true_img.shape[0], true_img.shape[1]
        # if verbose: print("Old Size : ", true_img.shape)
        #
        # img_dim_1, img_dim_2 = 0, 0
        #
        # if mode == "patch" and self.type_true_upscaling:
        #     # Overriding mode for True Upscaling models
        #     mode = 'fast'
        #     print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")
        #
        # if mode == 'patch':
        #     # Create patches
        #     if self.type_requires_divisible_shape:
        #         if patch_size % 4 != 0:
        #             print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
        #             patch_size = 8
        #
        #     images = img_utils.make_patches(true_img, scale_factor, patch_size, verbose=verbose)
        #
        #     nb_images = images.shape[0]
        #     img_dim_1, img_dim_2 = images.shape[1], images.shape[2]
        #     print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_dim_2, img_dim_1))
        # else:
        #     # Use full image for super resolution
        #     img_dim_1, img_dim_2 = self.__match_autoencoder_size(img_dim_1, img_dim_2, init_dim_1, init_dim_2,
        #                                                          scale_factor)
        #
        #     images = resize(true_img, (img_dim_1, img_dim_2))
        #     images = np.expand_dims(images, axis=0)
        #     print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))
        #
        # # Save intermediate bilinear scaled image is needed for comparison.
        # intermediate_img = None
        # if save_intermediate:
        #     if verbose: print("Saving intermediate image.")
        #     fn = path[0] + "_intermediate_" + path[1]
        #     intermediate_img = resize(true_img, (init_dim_1 * scale_factor, init_dim_2 * scale_factor))
        #     imwrite(fn, intermediate_img)

        # print("Images shape: ", images.shape)
        # # Transpose and Process images
        # # if K.image_dim_ordering() == "th":
        # #     img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        # # else:
        # #     img_conv = images.astype(np.float32) / 255.
        # img_conv = images.transpose((0, 2, 1, 3)).astype(np.float32) / 255.

        img_conv, h, w = self.__read_conv_img(img_path)
        img_conv = img_conv.transpose((0, 2, 1, 3)).astype(np.float32) / 255.

        model = self.create_model(h, w, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

        # Deprocess patches
        # if K.image_dim_ordering() == "th":
        #     result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        # else:
        #     result = result.astype(np.float32) * 255.
        result = result.transpose((0, 2, 1, 3)).astype(np.float32) * 255.

        result = result[0, :, :, :]  # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if _cv2_available:
            # used to remove noisy edges
            result = cv2.pyrUp(result)
            result = cv2.medianBlur(result, 3)
            result = cv2.pyrDown(result)

        if verbose: print("\nCompleted De-processing image.")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.", filename)
        cv2.imwrite(filename, result)

    def __read_conv_img(self, img_path):

        img_dir = os.path.dirname(img_path)
        files = []
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        for ext in exts:
            files.extend(glob.glob(os.path.join(img_dir, ext)))

        true_img = cv2.imread(img_path)
        h, w = true_img.shape[0], true_img.shape[1]

        X = np.zeros((1, h, w, 15))

        for img_idx, img_path in enumerate(files):
            img = cv2.imread(img_path)  # pilmode='RGB'
            img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
            j = 3 * img_idx

            X[0, :, :, j:(j + 3)] = img

        return X, h, w


class NonLocalResNetStitching(BaseSuperStitchingModel):

    def __init__(self):
        super(NonLocalResNetStitching, self).__init__("NonLocalResNetSR")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        # self.type_requires_divisible_shape = True
        # self.uses_learning_phase = False

        self.n = 32
        self.mode = 2

        self.weight_path = "weights/NonLocalResNetSR_Stitch.h5"
        # self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, nb_camera=5, load_weights=False):
        init = super(NonLocalResNetStitching, self).create_model(height, width, channels, nb_camera, load_weights)

        x0 = Convolution2D(self.n, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(init)
        x0 = non_local_block(x0)

        x = self._residual_block(x0, 1)

        nb_residual = 5
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = non_local_block(x, computation_compression=2)
        x = Add()([x, x0])

        # x = self._upscale_block(x, 1)

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)

        model = Model(init, x)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path, by_name=True)

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

        x = UpSampling2D()(init)
        x = Convolution2D(self.n, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)

        return x

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Non Local ResNetSR History.txt"):
        super(NonLocalResNetStitching, self).fit(batch_size, nb_epochs, save_history, history_fn)


class ImageStitchingModel(BaseSuperStitchingModel):

    def __init__(self):
        super(ImageStitchingModel, self).__init__("Image Stitching Model")

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Stitching Weights.h5"
        # self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, nb_camera=5, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ImageStitchingModel, self).create_model(height, width, channels, nb_camera, load_weights)

        x = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='level1')(init)
        x = Convolution2D(self.n2, (self.f2, self.f2), activation='relu', padding='same', name='level2')(x)

        out = Convolution2D(channels, (self.f3, self.f3), padding='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights:
            model.load_weights(self.weight_path)
        model.summary()

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="SRCNN History.txt"):
        return super(ImageStitchingModel, self).fit(batch_size, nb_epochs, save_history, history_fn)
