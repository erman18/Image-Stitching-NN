from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, \
    LeakyReLU
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from tensorflow.keras import backend as K
# from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers

from advanced import HistoryCheckpoint, non_local_block, TensorBoardBatch
import img_utils
from un_data_generator import un_read_img_dataset  # , DataGenerator, image_stitching_generator
import prepare_stitching_data as psd
from sklearn.model_selection import train_test_split
import constant as cfg

import numpy as np
import os
import glob
import time
import warnings

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# tf.compat.v1.disable_eager_execution()

try:
    import cv2
    _cv2_available = True
except:
    warnings.warn('Could not load opencv properly. This may affect the quality of output images.')
    _cv2_available = False

# Make Gaussian kernel following SciPy logic
## gauss_kernel = make_gaussian_2d_kernel(std, halfx)
def make_gaussian_2d_kernel(sigma, radius, dtype=tf.float32):
    # radius = tf.cast(sigma * truncate, tf.int32)
    x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
    k = tf.exp(-0.5 * tf.square(x / sigma))
    k = k / tf.reduce_sum(k)
    return tf.expand_dims(k, 1) * k

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
    # return K.cast(-10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.), dtype='float32')


def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                    str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


def SSIMLoss(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

# def custom_loss(y_true, y_pred):
#     '''
#     Loss functon for the Unsupervised Training
#         - y_true: is the same shape (W * H * 3 * N) as the input dataset
#             where N is the number of cameras/layers.
#         - y_pred: is the target stitched image (W * H * 3)
#     '''
#     print("Is eager excution enable: ", tf.executing_eagerly())
#     dy_pred, dx_pred = tf.image.image_gradients(y_pred)
#     # dy_true, dx_true = tf.image.image_gradients(y_true)

#     # Mask applied
#     m_loss = 0
#     for k in range(abs(cfg.sandfall_layer)):
#         y_true_k = y_true[:, :, :, 3*k:3*(k+1)]
#         # y_true_k = y_true[:, :, :, 0:3]
#         # tf.print("+++++++++++shape: ", y_true_k.shape, "\n", y_true.shape)
#         # dy_true_k = y_true_k[:, 1:, :, :] - y_true_k[:, :-1, :, :]
#         # dx_true_k = y_true_k[:, :, 1:, :] - y_true_k[:, :, :-1, :]
#         dy_true_k, dx_true_k = tf.image.image_gradients(y_true_k)

#         dx_pred1 = dx_pred[y_true_k != 0]
#         dy_pred1 = dy_pred[y_true_k != 0]

#         # Use convolution to compute the correlation between patches.
#         m_loss += K.mean(K.abs(dy_pred1 - dy_true_k) + K.abs(dx_pred1 - dx_true_k), axis=-1)
#         # masked_loss = np.abs(1 - y_pred[y_pred != 0]).mean()
#     return m_loss

# Keras losses
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

class gradient_layer_loss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.alpha = 0.08
        super(gradient_layer_loss, self).__init__(**kwargs)

    # def call(self, inputs):
    #     # X: input tensor
    #     # Out: output tensor
    #     inp, out = inputs
    #     dy_pred, dx_pred = tf.image.image_gradients(out)
    #     dy_true, dx_true = tf.image.image_gradients(inp)

    #     m_loss = 0
    #     for k in range(abs(cfg.sandfall_layer)):
            
    #         mask_x = tf.where(inp[:, :, :, 3*k:3*(k+1)] != 0, 1., 0.)
    #         mask_y = tf.where(inp[:, :, :, 3*k:3*(k+1)] != 0, 1., 0.)
    #         y_c = tf.abs((mask_y*dy_pred) - dy_true[:, :, :, 3*k:3*(k+1)])
    #         x_c = tf.abs((mask_x*dx_pred) - dx_true[:, :, :, 3*k:3*(k+1)])

    #         # kernels = np.expand_dims(np.ones((4,4,3), dtype=x_c.dtype), -2)
    #         kernels = np.ones((33,33,3,3), dtype=np.float32)
    #         kernels_tf = tf.constant(kernels, dtype=tf.float32)

    #         # Output tensor has shape [batch_size, h, w, d * num_kernels].
    #         strides = [1, 1, 1, 1]
    #         # output = tf.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    #         out_x = tf.nn.conv2d(x_c, kernels_tf, strides=strides, padding='SAME')
    #         out_y = tf.nn.conv2d(y_c, kernels_tf, strides=strides, padding='SAME')

    #         # Use convolution to compute the correlation between patches.
    #         # m_loss += K.mean(K.abs((mask_y*dy_pred) - dy_true[:, :, :, 3*k:3*(k+1)]) + K.abs((mask_x*dx_pred) - dx_true[:, :, :, 3*k:3*(k+1)]), axis=-1)
    #         m_loss += tf.reduce_mean(out_x + out_y, axis=None)

    #     return m_loss

    def call(self, inputs):
        # X: input tensor
        # Out: output tensor
        print("Is eager excution enable: ", tf.executing_eagerly())
        inp, out = inputs
        dy_pred, dx_pred = tf.image.image_gradients(out)
        dy_true, dx_true = tf.image.image_gradients(inp)

        # kernels = np.expand_dims(np.ones((4,4,3), dtype=x_c.dtype), -2)
        # kernels = np.ones((15,15,3,1), dtype=np.float32)
        kernels = np.ones((15,15,3,1), dtype=np.float32) # * np.array([0.2126, 0.7152, 0.0722])
        kernels_tf = tf.constant(kernels, dtype=tf.float32)

        # Output tensor has shape [batch_size, h, w, d * num_kernels].
        strides = [1, 1, 1, 1]

        dx_pred_c = tf.nn.conv2d(dx_pred, kernels_tf, strides=strides, padding='SAME')
        dy_pred_c = tf.nn.conv2d(dy_pred, kernels_tf, strides=strides, padding='SAME')

        m_loss = 0
        for k in range(abs(cfg.sandfall_layer)):
            
            mask_x = tf.where(inp[:, :, :, 3*k:3*(k+1)] != 0, 1., 0.)
            mask_y = tf.where(inp[:, :, :, 3*k:3*(k+1)] != 0, 1., 0.)

            # output = tf.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
            dx_true_c = tf.nn.conv2d(dx_true[:, :, :, 3*k:3*(k+1)], kernels_tf, strides=strides, padding='SAME')
            dy_true_c = tf.nn.conv2d(dy_true[:, :, :, 3*k:3*(k+1)], kernels_tf, strides=strides, padding='SAME')

            y_c = tf.abs((mask_y*dy_pred_c) - dy_true_c)
            x_c = tf.abs((mask_x*dx_pred_c) - dx_true_c)

            # Use convolution to compute the correlation between patches.
            # m_loss += K.mean(K.abs((mask_y*dy_pred) - dy_true[:, :, :, 3*k:3*(k+1)]) + K.abs((mask_x*dx_pred) - dx_true[:, :, :, 3*k:3*(k+1)]), axis=-1)
            m_loss += tf.reduce_mean(y_c + x_c, axis=None)
            # tf.keras.losses.Loss

        return self.alpha*m_loss

class BaseStitchingModel(object):

    def __init__(self, model_name):
        """
        Base model to provide a standard interface of adding Image Stiching models
        """
        self.shape = (None, None, None)
        self.model = None  # type: Model
        self.model_name = model_name
        self.weight_path = None


    def create_model(self, height=32, width=32, channels=3, nb_camera=5, load_weights=False) -> Model:
        """
        Subclass dependent implementation.
        """
        self.shape = (width, height, channels * nb_camera)

        init = Input(shape=self.shape)

        return init

    def fit(self, batch_size=32, nb_epochs=100, save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        """

        if self.model is None: self.create_model()

        # callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
        #                                            mode='max', save_weights_only=True, verbose=2)]
        callback_list = []

        # Parameters
        params = {'dim': (self.shape[0], self.shape[1]),
                  'batch_size': batch_size,
                  'n_channels': self.shape[2],
                  'shuffle': True}

        print("*************", self.shape[0], self.shape[1])
        # Datasets
        config_data = psd.read_json_file(cfg.config_img_output)
        data_indexes = np.arange(config_data["total_samples"])
        # samples_per_epoch = len(data_indexes)
        train_indexes, test_indexes = train_test_split(data_indexes, test_size=0.10)
        samples_per_epoch = len(train_indexes)
        val_count = len(test_indexes)

        training_generator = un_read_img_dataset(train_indexes, config_data, callee="un_training_generator", **params)
        validation_generator = un_read_img_dataset(test_indexes, config_data, callee="un_validation_generator", **params)
        callback_list.append(callbacks.ModelCheckpoint(self.weight_path, monitor='loss', save_best_only=True,
                                                mode='min', save_weights_only=True, verbose=2))

        # TODO: Guard this with an if condition to use this checkpoint only when dealing with supervised training
        callback_list.append(callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=2))

        if save_history:
            callback_list.append(HistoryCheckpoint(f'{cfg.log_dir}/{history_fn}'))

            if K.backend() == 'tensorflow':
                log_dir = f'{cfg.log_dir}/{self.model_name}_logs/'
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % self.__class__.__name__)

        self.model.fit(training_generator,
                       epochs=nb_epochs,
                       validation_data=validation_generator,
                       validation_steps=val_count // batch_size + 1,
                       steps_per_epoch=samples_per_epoch // batch_size + 1,
                       callbacks=callback_list,
                       use_multiprocessing=True,
                       workers=2)

        return self.model

    def simple_stitch(self, img_conv, out_dir, suffix=None, return_image=False, scale_factor=1, verbose=True):
        """
        Standard method to upscale an image.
        :param img_path: list of path to input images
        :param out_file: Output folder to save all the results
        :param suffix: Suffix the be added to the output filename
        :param return_image: returns a image of shape (height, width, channels).
        :param scale_factor: image scaled factor to resize input images
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """

        filename = os.path.join(out_dir, "result_" + str(suffix) +
                                time.strftime("_%Y%m%d-%H%M%S") + ".jpg")
        print("Output Result File: %s" % filename)
        os.makedirs(out_dir, exist_ok=True)

        # img_conv, h, w = self.__read_conv_img(img_path, scale_factor)
        h, w = img_conv.shape[1], img_conv.shape[2]
        img_conv = img_conv.transpose((0, 2, 1, 3))  # .astype(np.float32)

        print("Convolution image data point ready to be used: ", img_conv.shape)

        model = self.create_model(height=h, width=w, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        print("Starting the image stitching prediction")
        result = model.predict(img_conv, verbose=verbose, workers=2, use_multiprocessing=True)

        # Deprocess patches
        if verbose: print("De-processing images.")

        result = result.transpose((0, 2, 1, 3)).astype(np.float32) * 255.

        result = result[0, :, :, :]  # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if _cv2_available:
            # used to remove noisy edges
            result = cv2.pyrUp(result)
            result = cv2.medianBlur(result, 3)
            result = cv2.pyrDown(result)

        if verbose: print("\nCompleted De-processing image.")

        if verbose: print("Saving image.", filename)
        # Convert into BGR to save with OpenCV
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, result)

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result


class ResNetStitch(BaseStitchingModel):

    def __init__(self):
        super(ResNetStitch, self).__init__("UnResNetStitch")

        self.n = 64
        # self.mode = 2

        self.weight_path = "weights/UnResNetStitch.h5"

    def create_model(self, height=32, width=32, channels=3, nb_camera=5, load_weights=False):
        init = super(ResNetStitch, self).create_model(height, width, channels, nb_camera, load_weights)

        x0 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1', kernel_initializer="he_normal")(init)

        x = self._residual_block(x0, 1)

        nb_residual = 5
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = Add()([x, x0])

        # x = self._upscale_block(x, 1)
        # x = Add()([x, x1])

        # x = self._upscale_block(x, 2)
        # x = Add()([x, x0])

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final', kernel_initializer="he_normal")(x)

        m_custom_loss = gradient_layer_loss()([init, x])
        model = Model(init, x)
        # model.build(self.shape)

        adam = optimizers.Adam(learning_rate=1e-3)
        model.add_loss(m_custom_loss)
        # potential solution to explore: Check VAE Loss

        model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss, SSIMLoss])

        if load_weights and os.path.exists(self.weight_path):
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path, by_name=True)
        elif load_weights:
            cfg.PRINT_WARNING(f"Cannot load the file {self.weight_path}, it doesn't exist!")
        model.summary()

        self.model = model
        return model

    def _residual_block(self, ip, id):
        # mode = True # False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1', kernel_initializer="he_normal")(ip)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x)
        # x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)
        x = LeakyReLU(alpha=0.2, name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2', kernel_initializer="he_normal")(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x)
        # x = LeakyReLU(alpha=0.2, name="sr_res_activation_" + str(id) + "_2")(x)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    # def _upscale_block(self, ip, id):
    #     init = ip

    #     channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    #     channels = init.shape[channel_dim]  # init._keras_shape[channel_dim]

    #     # x = Convolution2D(256, (3, 3), activation="relu", padding='same', name='sr_res_upconv1_%d' % id)(init)
    #     # x = SubPixelUpscaling(r=2, channels=self.n, name='sr_res_upscale1_%d' % id)(x)
    #     x = UpSampling2D()(init)
    #     x = Convolution2D(self.n, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)

    #     # x = Convolution2DTranspose(channels, (4, 4), strides=(2, 2), padding='same', activation='relu',
    #     #                            name='upsampling_deconv_%d' % id)(init)

    #     return x

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="ResNetSR History.txt"):
        super(ResNetStitch, self).fit(batch_size, nb_epochs, save_history, history_fn)

