import numpy as np
import keras
import prepare_stitching_data as psd
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
        self.scene_range = []
        end_range = 0
        for idx in range(self.config_data["total_scene"]):
            end_range += self.config_data[str(idx)]["patchX"] * self.config_data[str(idx)]["patchY"]
            self.scene_range.append(end_range)
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

    def __get_scene_patch(self, index):
        # Find scene id
        scene_id = -1
        adj_idx = index

        for sc_id, end_r in enumerate(self.scene_range):
            if index < end_r:
                scene_id = sc_id
                if sc_id >= 1:
                    adj_idx = index - self.scene_range[sc_id - 1]
                break

        if scene_id < 0:
            raise ValueError("Invalid index %d" % index)

        # print("Adjusted index", adj_idx)
        total_imgs = self.config_data[str(scene_id)]["nb_imgs"]
        patchx_id, patchy_id = divmod(adj_idx, self.config_data[str(scene_id)]["patchY"])
        return scene_id, patchx_id, patchy_id, total_imgs

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
            scene_id, patchx_id, patchy_id, total_imgs = self.__get_scene_patch(ID)

            for img_idx in range(total_imgs):
                img_path = psd.training_folder + "/X/camID%d/imgID%d" % (scene_id, img_idx)
                img_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
                # print("Image path: %s" % img_path)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Failed to read the image: %s" % img_path)
                # cv2.imshow('image', img)
                img = img.astype('float32') / 255.

                j = 3 * img_idx

                X[i, :, :, j:(j + 3)] = img

            img_target_path = psd.training_folder + "/Y/camID%d" % scene_id
            img_target_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
            # print("target image path: %s" % img_target_path)
            target_img = cv2.imread(img_target_path, cv2.IMREAD_COLOR)
            target_img = target_img.astype('float32') / 255.
            y[i,] = target_img

        return X, y

    def test_get_scene_patch(self, index):
        return self.__get_scene_patch(index)

    def test_data_generator(self, index_list):
        return self.__data_generation(index_list)


def image_stitching_generator(list_ids, config_data, callee=None, batch_size=32, dim=(256, 256),
                              n_channels=15, shuffle=True, seed=None):

    def __get_scene_patch(index):
        # Find scene id
        scene_id = -1
        adj_idx = index

        for sc_id, end_r in enumerate(scene_range):
            if index < end_r:
                scene_id = sc_id
                if sc_id >= 1:
                    adj_idx = index - scene_range[sc_id - 1]
                break

        if scene_id < 0:
            raise ValueError("Invalid index %d" % index)

        # print("Adjusted index", adj_idx)
        total_imgs = config_data[str(scene_id)]["nb_imgs"]
        patchx_id, patchy_id = divmod(adj_idx, config_data[str(scene_id)]["patchY"])
        return scene_id, patchx_id, patchy_id, total_imgs

    # Estimate the range of ids for each scene
    scene_range = []
    end_range = 0
    for idx in range(config_data["total_scene"]):
        end_range += config_data[str(idx)]["patchX"] * config_data[str(idx)]["patchY"]
        scene_range.append(end_range)

    nb_images = len(list_ids)
    print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)
        # print("-----------image_shape: ", image_shape, "- current_batch_size", current_batch_size,
        #       "- y_image_shape", y_image_shape, "- _image_scale_multiplier: ", _image_scale_multiplier,
        #       "- small_train_images: ", small_train_images)

        # Find list of IDs
        list_ids_temp = [list_ids[k] for k in index_array]

        # Generate data
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        print("[%s]: Len of list_ids_temp: %d" % (callee, len(list_ids_temp)))
        batch_x = np.zeros((current_batch_size, *dim, n_channels))
        batch_y = np.zeros((current_batch_size, *dim, 3))

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            scene_id, patchx_id, patchy_id, total_imgs = __get_scene_patch(ID)

            for img_idx in range(total_imgs):
                img_path = psd.training_folder + "/X/camID%d/imgID%d" % (scene_id, img_idx)
                img_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
                # print("Image path: %s" % img_path)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = img.astype('float32') / 255.

                j = 3 * img_idx
                batch_x[i, :, :, j:(j + 3)] = img

            img_target_path = psd.training_folder + "/Y/camID%d" % scene_id
            img_target_path += "/patchID_%d_%d.png" % (patchx_id, patchy_id)
            # print("target image path: %s" % img_target_path)
            target_img = cv2.imread(img_target_path, cv2.IMREAD_COLOR)
            target_img = target_img.astype('float32') / 255.
            batch_y[i, ] = target_img

        yield batch_x, batch_y

        gc.collect()
        keras.backend.clear_session()


def _index_generator(N, batch_size=32, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        if batch_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

        current_index = (batch_index * batch_size) % N

        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)


if __name__ == "__main__":

    import prepare_stitching_data as psd

    config_data = psd.read_json_file(psd.config_file)

    d = DataGenerator([], config_data)
    for i in range(12387):
        d.test_get_scene_patch(i)
    print(d.test_get_scene_patch(45 * 15))

    X, y = d.test_data_generator([23, 54, 12300])
    print("x.shape: ", X.shape)
    print("y.shape: ", y.shape)

    from sklearn.model_selection import train_test_split

    x = np.arange(12387)
    x_train, x_test = train_test_split(x, test_size=0.20)
    print(len(x_test))
    print(len(x_train))
