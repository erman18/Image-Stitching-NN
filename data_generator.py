import os.path

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

import prepare_stitching_data as psd
import constant as cfg
import cv2
import gc


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_ids, config_data, callee=None, batch_size=32, dim=(256, 256), n_channels=15, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.callee = callee
        self.config_data = config_data  # psd.read_json_file(psd.config_file)
        self.dataset_index_range = []
        self.total_patch_per_img = []
        end_range = 0
        for idx in range(self.config_data["total_dataset"]):
            nb_patch_per_img = self.config_data[str(idx)]["patchX"] * \
                                 self.config_data[str(idx)]["patchY"]
            end_range += self.config_data[str(idx)]["nb_imgs"] * nb_patch_per_img

            self.total_patch_per_img.append(nb_patch_per_img)
            self.dataset_index_range.append(end_range)

        self.indexes = np.arange(len(self.list_ids))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    # @profile
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.shuffle = False
        gc.collect()
        keras.backend.clear_session()

    def __get_dataset_patch(self, index):
        # Find scene id
        dataset_id = -1
        adj_idx = index

        for sc_id, end_r in enumerate(self.dataset_index_range):
            if index < end_r:
                dataset_id = sc_id
                if sc_id >= 1:
                    adj_idx = index - self.dataset_index_range[sc_id - 1]
                break

        if dataset_id < 0:
            raise ValueError("Invalid index %d" % index)

        # print("Adjusted index", adj_idx)
        total_imgs = self.config_data[str(dataset_id)]["nb_imgs"]

        img_idx, patch_id = divmod(adj_idx, self.total_patch_per_img[dataset_id])
        patchx_id, patchy_id = divmod(patch_id, self.config_data[str(dataset_id)]["patchY"])

        # img_idx, patch_id = divmod(adj_idx, self.config_data[str(dataset_id)]["patchX"])
        # patchx_id, patchy_id = divmod(patch_id, self.config_data[str(dataset_id)]["patchY"])

        return dataset_id, patchx_id, patchy_id, img_idx, total_imgs

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size, *self.dim, 3))
        print("[%s]: Len of list_ids_temp: %d" % (self.callee, len(list_ids_temp)))
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.dim, 3))

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            dataset_id, patchx_id, patchy_id, img_idx, total_imgs = self.__get_dataset_patch(ID)
            ts = psd.TrainingSample(datasetID=dataset_id, imgID=img_idx, patchX=patchx_id,
                                    patchY=patchy_id, image_folder=cfg.image_folder)

            if os.path.exists(ts.get_sample_path()):
                print(f"==> index: {ID}, {ts.get_sample_path()}, check: {os.path.exists(ts.get_sample_path())}")
            X0 = ts.load_sample()
            y0 = ts.load_target()

            X[i, ] = X0

            # for img_idx in range(total_imgs):
            #     # ts = psd.TrainingSample(0, img_idx, patchX=patchx_id, patchY=patchy_id,)
            #     img_path = cfg.image_folder + "/X/camID%d/imgID%d" % (dataset_id, img_idx)
            #     img_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
            #     # print("Image path: %s" % img_path)
            #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            #     if img is None:
            #         raise ValueError("Failed to read the image: %s" % img_path)
            #     # cv2.imshow('image', img)
            #     img = img.astype('float32') / 255.
            #
            #     j = 3 * img_idx
            #
            #     X[i, :, :, j:(j + 3)] = img

            # img_target_path = cfg.image_folder + "/Y/camID%d" % dataset_id
            # img_target_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
            # # print("target image path: %s" % img_target_path)
            # target_img = cv2.imread(img_target_path, cv2.IMREAD_COLOR)
            # target_img = target_img.astype('float32') / 255.
            # y[i,] = target_img
            y[i,] = y0

        return X, y

    def __img_path_generation(self, list_ids):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization

        X = []

        # Generate data
        for i, ID in enumerate(list_ids):
            # Store sample
            dataset_id, patchx_id, patchy_id, img_id, total_imgs = self.__get_dataset_patch(ID)
            X.append((dataset_id, patchx_id, patchy_id, img_id, total_imgs))

            # path_samples = []
            #
            # for img_idx in range(total_imgs):
            #     img_path = psd.training_folder + "/X/camID%d/imgID%d/patchID_%d_%d.png" \
            #                % (scene_id, img_idx, patchx_id, patchy_id)
            #
            #     path_samples.append(img_path)
            #
            # img_target_path = psd.training_folder + "/Y/camID%d/patchID_%d_%d.png" % (scene_id, patchx_id, patchy_id)
            #
            # # X.append((path_samples, img_target_path))
            # path_samples.append(img_target_path)
            # X.append(path_samples)

        return X

    def test_get_scene_patch(self, index):
        return self.__get_dataset_patch(index)

    def test_data_generator(self, index_list):
        # Store sample
        paths = []
        for ID in index_list:
            dataset_id, patchx_id, patchy_id, img_idx, total_imgs = self.__get_dataset_patch(ID)
            ts = psd.TrainingSample(datasetID=dataset_id, imgID=img_idx, patchX=patchx_id,
                                patchY=patchy_id, image_folder=cfg.image_folder)

            paths.append(ts.get_sample_path())
            if not os.path.exists(ts.get_sample_path()):
                print(f"==> index: {ID}, {ts.get_sample_path()}, check: {os.path.exists(ts.get_sample_path())}")

        print("==> Checking for duplicate")
        if len(paths) != len(set(paths)):
            print("There are duplicates.")
        else:
            print("All file names are unique")

        print("test_data_generator Done!")
        return

    def test_path_generator(self, index_list):
        return self.__img_path_generation(index_list)

    def generate_img_path(self):
        return self.__img_path_generation(self.list_ids)


# def image_stitching_generator(list_ids, config_data, callee=None, batch_size=32, dim=(256, 256),
#                               n_channels=15, shuffle=True, seed=None):
#     def __get_scene_patch(index):
#         # Find scene id
#         scene_id = -1
#         adj_idx = index
#
#         for sc_id, end_r in enumerate(scene_range):
#             if index < end_r:
#                 scene_id = sc_id
#                 if sc_id >= 1:
#                     adj_idx = index - scene_range[sc_id - 1]
#                 break
#
#         if scene_id < 0:
#             raise ValueError("Invalid index %d" % index)
#
#         # print("Adjusted index", adj_idx)
#         total_imgs = config_data[str(scene_id)]["nb_imgs"]
#         patchx_id, patchy_id = divmod(adj_idx, config_data[str(scene_id)]["patchY"])
#         return scene_id, patchx_id, patchy_id, total_imgs
#
#     # Estimate the range of ids for each scene
#     scene_range = []
#     end_range = 0
#     for idx in range(config_data["total_dataset"]):
#         end_range += config_data[str(idx)]["patchX"] * config_data[str(idx)]["patchY"]
#         scene_range.append(end_range)
#
#     nb_images = len(list_ids)
#     print("Found %d images." % nb_images)
#
#     index_generator = _index_generator(nb_images, batch_size, shuffle, seed)
#
#     while 1:
#         index_array, current_index, current_batch_size = next(index_generator)
#         # print("-----------image_shape: ", image_shape, "- current_batch_size", current_batch_size,
#         #       "- y_image_shape", y_image_shape, "- _image_scale_multiplier: ", _image_scale_multiplier,
#         #       "- small_train_images: ", small_train_images)
#
#         # Find list of IDs
#         list_ids_temp = [list_ids[k] for k in index_array]
#
#         # Generate data
#         """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
#         # Initialization
#         print("[%s]: Len of list_ids_temp: %d" % (callee, len(list_ids_temp)))
#         batch_x = np.zeros((current_batch_size, *dim, n_channels))
#         batch_y = np.zeros((current_batch_size, *dim, 3))
#
#         # Generate data
#         for i, ID in enumerate(list_ids_temp):
#             # Store sample
#             scene_id, patchx_id, patchy_id, total_imgs = __get_scene_patch(ID)
#
#             for img_idx in range(total_imgs):
#                 img_path = psd.training_folder + "/X/camID%d/imgID%d" % (scene_id, img_idx)
#                 img_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
#                 # print("Image path: %s" % img_path)
#                 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#                 img = img.astype('float32') / 255.
#
#                 j = 3 * img_idx
#                 batch_x[i, :, :, j:(j + 3)] = img
#
#             img_target_path = psd.training_folder + "/Y/camID%d" % scene_id
#             img_target_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
#             # print("target image path: %s" % img_target_path)
#             target_img = cv2.imread(img_target_path, cv2.IMREAD_COLOR)
#             target_img = target_img.astype('float32') / 255.
#             batch_y[i,] = target_img
#
#         yield batch_x, batch_y
#
#         gc.collect()
#         keras.backend.clear_session()
#
#
# def _index_generator(N, batch_size=32, shuffle=True, seed=None):
#     batch_index = 0
#     total_batches_seen = 0
#     index_array = []
#
#     while 1:
#         if seed is not None:
#             np.random.seed(seed + total_batches_seen)
#
#         if batch_index == 0:
#             index_array = np.arange(N)
#             if shuffle:
#                 index_array = np.random.permutation(N)
#
#         current_index = (batch_index * batch_size) % N
#
#         if N >= current_index + batch_size:
#             current_batch_size = batch_size
#             batch_index += 1
#         else:
#             current_batch_size = N - current_index
#             batch_index = 0
#         total_batches_seen += 1
#
#         yield (index_array[current_index: current_index + current_batch_size],
#                current_index, current_batch_size)
#
#
# @tf.function
# def load_image(image_path_data, dim, n_channels, base_dir):
#
#     scene_id = image_path_data[0]
#     patchx_id = image_path_data[1]
#     patchy_id = image_path_data[2]
#     total_imgs = image_path_data[3]
#
#     max_seq_len = n_channels // 3
#
#     X = tf.TensorArray(tf.float32, size=max_seq_len)
#
#     for id in tf.range(total_imgs):
#         image_path = tf.strings.format("{}/X/camID{}/imgID{}/patchID_{}_{}.png",
#                                        (base_dir, scene_id, patchx_id, patchy_id, id))
#
#         img = tf.image.decode_png(tf.io.read_file(image_path), channels=3)
#         # img = tf.image.resize(img, (299, 299))
#         img = tf.cast(img, tf.float32) / 255.0
#         # X.append(img)
#         X = X.write(id, img)
#
#     for id in tf.range(max_seq_len - total_imgs):
#         img = tf.zeros(shape=[*dim, 3], dtype=tf.float32)
#         X = X.write(id, img)
#
#     target_path = tf.strings.format("{}/Y/camID{}/patchID_{}_{}.png", (base_dir, scene_id, patchx_id, patchy_id))
#     y = tf.image.decode_png(tf.io.read_file(target_path), channels=3)
#     y = tf.cast(y, tf.float32) / 255.0
#
#     return X.stack(), y


def load_data_sample(img_paths, dim, n_channels, training_folder):

    dataset_id, patchx_id, patchy_id, img_id, total_imgs = img_paths

    ts = psd.TrainingSample(datasetID=dataset_id, imgID=img_id, patchX=patchx_id,
                            patchY=patchy_id, image_folder=training_folder.decode("utf-8"))
    X = ts.load_sample()
    y = ts.load_target()

    # X = np.zeros((*dim, n_channels), dtype=np.float32)
    # for img_idx in range(total_imgs):
    #     # img_path = f'{training_folder.decode("utf-8")}/X/camID{dataset_id}' \
    #     #            f'/imgID{img_idx}/patchID_{patchx_id}_{patchy_id}.png'
    #     # # print(img_path)
    #     #
    #     # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     # img = img.astype('float32') / 255.
    #     #
    #     # j = 3 * img_idx
    #     # X[:, :, j:(j + 3)] = img
    #
    # img_target_path = f'{training_folder.decode("utf-8")}/Y/camID{dataset_id}/patchID_{patchx_id}_{patchy_id}.png'
    #
    # target_img = cv2.imread(img_target_path, cv2.IMREAD_COLOR)
    # y = target_img.astype('float32') / 255.

    return X, y


# @tf.function
def read_img_dataset(list_ids, config_data, callee, batch_size=32, dim=(256, 256),
                     n_channels=15, shuffle=True, seed=None, buffer_size=2):

    print("callee: %s" % callee, len(list_ids))
    data_gen = DataGenerator(list_ids, config_data, dim=dim)
    list_paths = data_gen.generate_img_path()

    dataset = tf.data.Dataset.from_tensor_slices(list_paths)
    # dataset = dataset.map(load_image)

    # dataset = dataset.map(lambda x: load_image(x, dim, n_channels, psd.training_folder))

    dataset = dataset.map(lambda item1, item2=dim, item3=n_channels, item4=str(cfg.image_folder): tf.numpy_function(
        load_data_sample, [item1, item2, item3, item4], [tf.float32, tf.float32]))  #

    return dataset.repeat(1).batch(batch_size=batch_size).prefetch(buffer_size)


if __name__ == "__main__":

    config_data = psd.read_json_file(cfg.config_img_output)

    d = DataGenerator([], config_data, dim=(cfg.patch_size, cfg.patch_size))
    # for i in range(12387):
    #     d.test_get_scene_patch(i)
    # print(d.test_get_scene_patch(45 * 15))

    # X, y = d.test_data_generator([0, 23, 54, 2050])
    # print("x.shape: ", X.shape)
    # print("y.shape: ", y.shape)

    d.test_data_generator([*range(config_data["total_samples"])])

    print("==> For consistency in the total_sample")
    print(f'total_samples: {config_data["total_samples"]}')

    # print("Len X: ", len(X), X)

    # from sklearn.model_selection import train_test_split
    #
    # x = np.arange(12387)
    # x_train, x_test = train_test_split(x, test_size=0.20)
    # print(len(x_test))
    # print(len(x_train))
