from __future__ import print_function, division

import glob
import json
import os
import numpy as np

from cv2 import imwrite, imread

import img_utils
import project_settings as cfg


def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError("duplicate key: %r" % (k,))
        else:
            d[k] = v
    return d


def write_json_file(filename, data=dict()):
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


dataset_folder = cfg.base_folder + "/images"
target_folder = cfg.base_folder + "/target"
training_folder = cfg.base_folder + "/train"
config_file = cfg.base_folder + "/config_file.json"


def prepare_data(step=32, patch_size=128):
    """Prepare data set for image stitching"""
    os.makedirs(training_folder, exist_ok=True)

    # Read all image in the folder (Required python >= 3.5)
    image_dir = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
    # target_dir = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]

    print(image_dir)

    data_file_settings = read_json_file(config_file)
    data_file_settings["total_scene"] = len(image_dir)
    data_file_settings["total_samples"] = 0

    for cam_idx, cam_dir in enumerate(image_dir):
        print(cam_dir)

        target_file = target_folder + "/" + os.path.basename(cam_dir) + ".png"
        print(target_file)

        files = []
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        for ext in exts:
            files.extend(glob.glob(os.path.join(cam_dir, ext)))

        # data_file_settings[str(cam_idx)] = {}
        # patch parameters
        # step = 32
        # patch_size = 128

        for img_idx, img_path in enumerate(files):
            img = imread(img_path)  # pilmode='RGB'
            img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

            img_patches = img_utils.make_raw_patches(img, step=step, patch_size=patch_size, verbose=1)

            print(img_patches.shape)
            img_dir = training_folder + "/X/camID%d/imgID%d" % (cam_idx, img_idx)
            print("======>", img_dir)

            imwrite(img_dir + ".png", img)

            os.makedirs(img_dir, exist_ok=True)
            for patchIdx in range(img_patches.shape[0]):
                for patchIdy in range(img_patches.shape[1]):
                    img_patch = img_patches[patchIdx, patchIdy, 0, :, :]

                    imwrite(img_dir + "/patchID_%d_%d.png" % (patchIdx, patchIdy), img_patch)

            data_file_settings[str(cam_idx)] = {
                "nbImg_per_scene": len(files),
                "nb_imgs": len(files),
                "patchX": img_patches.shape[0],
                "patchY": img_patches.shape[1],
                "patchSizeX": img_patches.shape[3],
                "patchSizeY": img_patches.shape[4],
            }

        data_file_settings["total_samples"] += data_file_settings[str(cam_idx)]["patchX"] * \
                                               data_file_settings[str(cam_idx)]["patchY"]

        target_img = imread(target_file)  # pilmode='RGB'
        target_patches = img_utils.make_raw_patches(target_img, step=step, patch_size=patch_size, verbose=1)
        target_out_dir = training_folder + "/Y/camID%d" % cam_idx
        os.makedirs(target_out_dir, exist_ok=True)
        for patchIdx in range(target_patches.shape[0]):
            for patchIdy in range(target_patches.shape[1]):
                img_patch = target_patches[patchIdx, patchIdy, 0, :, :]

                imwrite(target_out_dir + "/patchID_%d_%d.png" % (patchIdx, patchIdy), img_patch)

    print(data_file_settings)
    # print(data_file_settings.items())
    print(data_file_settings.keys())
    # Save configuration data
    write_json_file(config_file, data=data_file_settings)


if __name__ == "__main__":
    prepare_data()
    # os.makedirs(training_folder, exist_ok=True)
    #
    # # Read all image in the folder (Required python >= 3.5)
    # image_dir = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
    # # target_dir = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
    #
    # print(image_dir)
    #
    # data_file_settings = read_json_file(config_file)
    # data_file_settings["total_scene"] = len(image_dir)
    #
    # for cam_idx, cam_dir in enumerate(image_dir):
    #     print(cam_dir)
    #
    #     target_file = target_folder + "/" + os.path.basename(cam_dir) + ".png"
    #     print(target_file)
    #
    #     files = []
    #     exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    #     for ext in exts:
    #         files.extend(glob.glob(os.path.join(cam_dir, ext)))
    #
    #     # data_file_settings[str(cam_idx)] = {}
    #     # patch parameters
    #     step = 64
    #     patch_size = 256
    #
    #     for img_idx, img_path in enumerate(files):
    #         img = imread(img_path)  # pilmode='RGB'
    #         img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    #
    #         img_patches = img_utils.make_raw_patches(img, step=step, patch_size=patch_size, verbose=1)
    #
    #         print(img_patches.shape)
    #         img_dir = training_folder + "/X/camID%d/imgID%d" % (cam_idx, img_idx)
    #         print("======>", img_dir)
    #
    #         imwrite(img_dir + ".png", img)
    #
    #         os.makedirs(img_dir, exist_ok=True)
    #         for patchIdx in range(img_patches.shape[0]):
    #             for patchIdy in range(img_patches.shape[1]):
    #                 img_patch = img_patches[patchIdx, patchIdy, 0, :, :]
    #
    #                 imwrite(img_dir + "/patchID_%d_%d.png" % (patchIdx, patchIdy), img_patch)
    #
    #         data_file_settings[str(cam_idx)] = {
    #             "nbImg_per_scene": img_idx,
    #             "nb_imgs": len(files),
    #             "patchX": img_patches.shape[0],
    #             "patchY": img_patches.shape[1],
    #             "patchSizeX": img_patches.shape[3],
    #             "patchSizeY": img_patches.shape[4],
    #         }
    #
    #     target_img = imread(target_file)  # pilmode='RGB'
    #     target_patches = img_utils.make_raw_patches(target_img, step=step, patch_size=patch_size, verbose=1)
    #     target_out_dir = training_folder + "/Y/camID%d" % (cam_idx)
    #     os.makedirs(target_out_dir, exist_ok=True)
    #     for patchIdx in range(target_patches.shape[0]):
    #         for patchIdy in range(target_patches.shape[1]):
    #             img_patch = target_patches[patchIdx, patchIdy, 0, :, :]
    #
    #             imwrite(target_out_dir + "/patchID_%d_%d.png" % (patchIdx, patchIdy), img_patch)
    #
    # print(data_file_settings)
    # # print(data_file_settings.items())
    # print(data_file_settings.keys())
    # # Save configuration data
    # write_json_file(config_file, data=data_file_settings)
