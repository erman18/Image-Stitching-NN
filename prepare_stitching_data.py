from __future__ import print_function, division

import glob
import json
import os
import sys
import time

import numpy as np
from cv2 import imwrite, imread

import img_utils
import project_settings as cfg

# sys.path.append(cfg.pano_libs_dir)
import pylab as plt
import panowrapper as pw
from tqdm import tqdm


def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError("duplicate key: %r" % (k,))
        else:
            d[k] = v
    return d


def write_json_file(filename, data):
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4)


def read_json_file(filename):
    try:
        with open(filename, 'r') as fp:
            mdata = dict(json.load(fp, object_pairs_hook=dict_raise_on_duplicates))
    except IOError:
        print('File not found, will create a new one.')
        mdata = dict()

    return mdata


def read_json_arr(filename):
    try:
        with open(filename, 'r') as fp:
            mdata = json.loads(fp.read())
    except IOError:
        print('File not found, will create a new one.')

    return mdata


# dataset_folder = cfg.dataset_folder + "/images"
# target_folder = cfg.dataset_folder + "/target"
# training_folder = cfg.dataset_folder + "/train"
# config_file = cfg.dataset_folder + "/config_file.json"


class TrainingSample:
    def __init__(self, datasetID, imgID, patchX, patchY, image_folder):
        # self.sample_id = sampleID
        self.datasetID = datasetID
        self.img_id = imgID
        self.patch_x = patchX
        self.patch_y = patchY

        self.dataset_path = f"{image_folder}/{datasetID}"
        self.sample_folder = f"{self.dataset_path}/X"
        self.target_folder = f"{self.dataset_path}/y"

        self.sample_path = f"{self.sample_folder}/imgID{self.img_id}_patchID{self.patch_x}_{self.patch_y}"
        self.target_path = f"{self.target_folder}/imgID{self.img_id}_patchID{self.patch_x}_{self.patch_y}"

    def get_sample_path(self):
        return f"{self.sample_path}.npz"

    def get_target_path(self):
        return f"{self.target_path}.npz"

    def save_sample(self, data):
        # np.save(f"{self.sample_path}.npy", data)
        np.savez_compressed(self.get_sample_path(), data=data)

    def save_target(self, data):
        # np.save(f"{self.target_path}.npy", data)
        np.savez_compressed(self.get_target_path(), data=data)

    def load_sample(self):
        # np.load(f"{self.sample_path}.npy")
        return np.load(self.get_sample_path())["data"]

    def load_target(self):
        # return np.load(f"{self.target_path}.npy")
        return np.load(self.get_target_path())["data"]


class Dataset:
    def __init__(self, datasetID, total_img, nb_cameras, nb_img_generate, img_pattern,
                 image_folder, data_settings=None):
        """
        Class to Generate the dataset from Pano Stitcher
        :param datasetID:
        :param size_x: patch_x size
        :param size_y: patch_y size
        :param nb_cameras: total number of cameras for this dataset
        :param nb_img_generate: total number of image to generate
        :param total_img: total number of images for this dataset
        :param img_pattern: image pattern to retrieve image files on disk. This pattern is
            a python f-string and should contain the camera id and the image id fields
            Example: /media/sf_Data/data_stitching/Terrace/Input/{camID:05d}/{imgID:05d}.jpg
        :param image_folder: Training folder to store images
        :param data_settings: Default: None. Dictionary of the data setting for this dataset
        """
        self.dataset_id = datasetID
        self.total_img = total_img
        self.img_pattern = img_pattern
        self.nb_cameras = nb_cameras
        self.nb_img_generate = nb_img_generate
        self.dataset_path = f"{image_folder}/{datasetID}"
        self.target_img_path = f"{self.dataset_path}/target"
        self.sample_folder = f"{self.dataset_path}/X"
        self.target_path = f"{self.dataset_path}/y"

        self.dataset_settings = data_settings
        if "total_dataset" not in self.dataset_settings:
            self.dataset_settings.clear()
            self.dataset_settings["total_dataset"] = 0
            self.dataset_settings["total_samples"] = 0
            write_json_file(cfg.config_img_output, data=self.dataset_settings)

        if str(self.dataset_id) not in self.dataset_settings:
            self.dataset_settings[str(self.dataset_id)] = {
                "nb_imgs": 0,
                "patchX": 0,
                "patchY": 0,
                "patchSizeX": cfg.patch_size,
                "patchSizeY": cfg.patch_size,
            }
            self.dataset_settings["total_dataset"] += 1

        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.sample_folder, exist_ok=True)
        os.makedirs(self.target_path, exist_ok=True)
        os.makedirs(self.target_img_path, exist_ok=True)

    def __convert_mat_np(self, mat):
        p = np.array(mat, copy=False)
        p[p < 0] = 0  # Replace negative values with zeros
        p[p > 1] = 1  # Replace negative values with zeros
        return p

    def set_dataset_settings(self, key, value):
        self.dataset_settings[str(self.dataset_id)][key] = value

    def get_dataset_settings(self, key, default_value=None):
        return self.dataset_settings[str(self.dataset_id)].get(key, default_value)

    def increment_settings(self, key, value):
        if key in self.dataset_settings[str(self.dataset_id)]:
            self.dataset_settings[str(self.dataset_id)][key] += value
        else:
            self.dataset_settings[str(self.dataset_id)][key] = value

    def generate_dataset(self):

        panow = pw.PanoWrapper()

        for img_id in range(self.total_img):

            file_list = [self.img_pattern.format(camID=i, imgID=img_id) for i in range(self.nb_cameras)]

            print(file_list)
            try:
                panow.init_pano_stitcher(file_list, multi_band_blend=cfg.sandfall_layer)
                break
            except:
                print(f"Error: Cannot stitch image [{img_id}]")

        if not panow.is_pano_initialize():
            raise RuntimeError("Failed to find the projection parameters. Please add calibration "
                               "images in the same director as the images")

        import random
        sample_img_id = random.sample(range(self.total_img), self.nb_img_generate)

        for img_id in sample_img_id:
            file_list = [self.img_pattern.format(camID=i, imgID=img_id) for i in range(self.nb_cameras)]

            print(file_list)
            mat_merge = panow.build_pano(file_list, multi_band_blend=cfg.sandfall_layer)
            pmat_merge = self.__convert_mat_np(mat_merge)
            mat_target = panow.build_pano(file_list, multi_band_blend=0)
            pmat_target = self.__convert_mat_np(mat_target)
            panow.write_img(f'{self.target_img_path}/tgID{self.get_dataset_settings("nb_imgs")}.jpg', mat_target)

            self.write_sample(self.get_dataset_settings("nb_imgs"), pmat_merge, pmat_target, cfg.patch_step,
                              cfg.patch_size)

    def write_sample(self, img_id, img, target, step, patch_size):

        img_patches = img_utils.make_raw_patches(img, step=step, patch_size=patch_size, channels=img.shape[-1],
                                                 verbose=1)
        target_patches = img_utils.make_raw_patches(target, step=step, patch_size=patch_size, channels=3, verbose=1)

        print(img_patches.shape)
        self.set_dataset_settings("patchX", img_patches.shape[0])
        self.set_dataset_settings("patchY", img_patches.shape[1])
        self.increment_settings("nb_imgs", 1)
        self.dataset_settings["total_samples"] += img_patches.shape[0]*img_patches.shape[1]

        for patchIdx in tqdm(range(img_patches.shape[0])):
            for patchIdy in tqdm(range(img_patches.shape[1]), leave=False):
                img_patch = img_patches[patchIdx, patchIdy, 0, :, :]
                target_patch = target_patches[patchIdx, patchIdy, 0, :, :]

                train_sample_obj = TrainingSample(datasetID=self.dataset_id, imgID=img_id, patchX=patchIdx,
                                                  patchY=patchIdy, image_folder=cfg.image_folder)

                train_sample_obj.save_sample(img_patch)
                train_sample_obj.save_target(target_patch)

        # Update config file each iteration
        write_json_file(cfg.config_img_output, data=self.dataset_settings)
        return


def prepare_data_live():
    """Prepare data set for image stitching"""

    data_file_settings = read_json_file(cfg.config_img_input)

    config_output_file = read_json_file(cfg.config_img_output)

    print(data_file_settings)
    for datasetID, dataset_dc in data_file_settings.items():
        print("datasetID", datasetID, "dataset_dc", dataset_dc)

        total_img = dataset_dc.get("total_img", 0)
        nb_cameras = dataset_dc.get("nb_cameras", 0)
        nb_img_generate = dataset_dc.get("nb_img_generate", 0)
        img_pattern = dataset_dc.get("img_pattern", None)
        image_folder = cfg.image_folder
        print("==>", img_pattern)
        ds = Dataset(datasetID, total_img, nb_cameras, nb_img_generate, img_pattern, image_folder,
                     config_output_file)
        ds.generate_dataset()

        # config_output_file[str(datasetID)] = ds.data_settings
        # config_output_file["total_samples"] += (ds.data_settings["nb_imgs"]
        #                                                         *ds.data_settings["patchX"]
        #                                                         *ds.data_settings["patchY"])

        # Update config file each iteration
        # write_json_file(cfg.config_img_output, data=data_file_settings)


# def prepare_data(step=cfg.patch_step, patch_size=cfg.patch_size, invert_white=False):
#     """Prepare data set for image stitching"""
#     os.makedirs(training_folder, exist_ok=True)
#
#     # Read all image in the folder (Required python >= 3.5)
#     image_dir = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
#     # target_dir = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
#
#     print(image_dir)
#
#     data_file_settings = read_json_file(config_file)
#     data_file_settings["total_dataset"] = len(image_dir)
#     data_file_settings["total_samples"] = 0
#
#     for cam_idx, cam_dir in enumerate(image_dir):
#         print(cam_dir)
#
#         target_file = target_folder + "/" + os.path.basename(cam_dir) + ".png"
#         print(target_file)
#
#         files = []
#         exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
#         for ext in exts:
#             files.extend(glob.glob(os.path.join(cam_dir, ext)))
#
#         # data_settings[str(cam_idx)] = {}
#         # patch parameters
#         # step = 32
#         # patch_size = 128
#
#         for img_idx, img_path in enumerate(files):
#             img = imread(img_path)  # pilmode='RGB'
#             if invert_white: img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
#
#             img_patches = img_utils.make_raw_patches(img, step=step, patch_size=patch_size, verbose=1)
#
#             print(img_patches.shape)
#             img_dir = training_folder + "/X/camID%d/imgID%d" % (cam_idx, img_idx)
#             print("======>", img_dir)
#
#             imwrite(img_dir + ".png", img)
#
#             os.makedirs(img_dir, exist_ok=True)
#             for patchIdx in range(img_patches.shape[0]):
#                 for patchIdy in range(img_patches.shape[1]):
#                     img_patch = img_patches[patchIdx, patchIdy, 0, :, :]
#
#                     imwrite(img_dir + "/patchID_%d_%d.png" % (patchIdx, patchIdy), img_patch)
#
#             data_file_settings[str(cam_idx)] = {
#                 "nbImg_per_scene": len(files),
#                 "nb_imgs": len(files),
#                 "patchX": img_patches.shape[0],
#                 "patchY": img_patches.shape[1],
#                 "patchSizeX": img_patches.shape[3],
#                 "patchSizeY": img_patches.shape[4],
#             }
#
#         data_file_settings["total_samples"] += data_file_settings[str(cam_idx)]["patchX"] * \
#                                                data_file_settings[str(cam_idx)]["patchY"]
#
#         target_img = imread(target_file)  # pilmode='RGB'
#         target_patches = img_utils.make_raw_patches(target_img, step=step, patch_size=patch_size, verbose=1)
#         target_out_dir = training_folder + "/Y/camID%d" % cam_idx
#         os.makedirs(target_out_dir, exist_ok=True)
#         for patchIdx in range(target_patches.shape[0]):
#             for patchIdy in range(target_patches.shape[1]):
#                 img_patch = target_patches[patchIdx, patchIdy, 0, :, :]
#
#                 imwrite(target_out_dir + "/patchID_%d_%d.png" % (patchIdx, patchIdy), img_patch)
#
#     print(data_file_settings)
#     # print(data_settings.items())
#     print(data_file_settings.keys())
#     # Save configuration data
#     write_json_file(config_file, data=data_file_settings)


if __name__ == "__main__":
    prepare_data_live()

    # import libpyopenpano as pano
    # # help(pano)
    # # Test Stitching
    # pano.print_config()
    # pano.init_config(cfg.pano_config_file)
    # pano.print_config()
    #
    # mdata = [
    #     {
    #         "img_pattern": "/media/sf_Data/data_stitching/Terrace/Input/{camID:05d}/{imgID:05d}.jpg",
    #         "out_dir": "/media/sf_Data/data_stitching/Terrace/Out2",
    #         "nb_cameras": 14,
    #         "calib_img_id": 0,
    #         "total_img": 430,
    #     },
    #     {
    #         "img_pattern": "/media/sf_Data/data_stitching/Terrace/Input/{camID:05d}/{imgID:05d}.jpg",
    #         "out_dir": "/media/sf_Data/data_stitching/Terrace/Out2",
    #         "nb_cameras": 14,
    #         "calib_img_id": 0,
    #         "total_img": 430,
    #     }
    # ]
    #
    # id = 0
    # # img_dir = "/media/sf_Data/data_stitching/Terrace/Input/{:05d}/{:05d}.jpg"
    # out_dir = mdata[id]["out_dir"]
    # nb_camera = mdata[id]["nb_cameras"]
    # total_img = mdata[id]["total_img"]
    # os.makedirs(out_dir, exist_ok=True)
    # output_result = None
    #
    # nb_stitched_img = 0
    # stitcher = None
    # for img_id in range(total_img):
    #     print(f"-----------------------------{img_id}------------------------")
    #     file_list = [mdata[id]["img_pattern"].format(camID=i, imgID=img_id) for i in range(mdata[id]["nb_cameras"])]
    #     output_result = f"{out_dir}/{img_id:05d}.jpg"
    #
    #     print(file_list)
    #     print(output_result)
    #
    #     stitcher = None
    #     try:
    #         stitcher = pano.Stitcher(file_list)
    #         mat = stitcher.build()
    #         pano.write_img(output_result, mat)
    #         break
    #     except:
    #         print(f"Error: Cannot stitch image [{img_id}] - [{output_result}]")
    #
    # print(f"First image stitched: {stitcher} --> Location: {output_result}")
    # multi_band_blend = 0  # 0 is for linear blending
    # time.sleep(10)
    # for img_id in range(total_img):
    #     print(f"-----------------------------{img_id}------------------------")
    #     file_list = [mdata[id]["img_pattern"].format(i, img_id) for i in range(mdata[id]["nb_cameras"])]
    #     output_result = f"{out_dir}/{img_id:05d}.jpg"
    #
    #     try:
    #         print(f"Try to build from new images {img_id}/{total_img}")
    #         # print(file_list)
    #         mat = stitcher.build_from_new_images(file_list, multi_band_blend)
    #         print(f"Done building from new images {img_id}/{total_img}")
    #         # time.sleep(10)
    #         pano.write_img(output_result, mat)
    #     except:
    #         print(f"[build_from_new_images] Error: Cannot stitch image [{img_id}] - [{output_result}]")
    #
    # # pano.test_extrema(mat, 1)
    # print("done stitching!", nb_stitched_img)
    # # pano.print_config()
    #
    # # p = np.array(mat, copy=False)
    # # plt.imshow(p)
    # # plt.show()
