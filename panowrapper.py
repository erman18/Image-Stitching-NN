from __future__ import print_function, division

import os
import sys
import time

import numpy as np

import project_settings as cfg
import cv2

sys.path.append(cfg.pano_libs_dir)

try:
    import libpyopenpano as pano
except:
    ValueError("Couldn't import 'libpyopenpano' library. You may need to use the shell "
               "script (*.sh files) to run this module or export LD_LIBRARY_PATH variable.\n"
               "    => Ex: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR && python prepare_stitching_data.py")


class PanoWrapper:

    def __init__(self, scale_factor=1.0, verbose=0):
        pano.init_config(cfg.pano_config_file)
        self.verbose = verbose
        self.scale_factor = scale_factor
        self.pano_stitch = None
        self.pano_stitch_init = False

        if self.verbose:
            pano.print_config()

    def print_config(self):
        pano.print_config()

    def is_pano_initialize(self):
        return self.pano_stitch_init

    def init_pano_stitcher(self, calib_files, multi_band_blend):
        pano_stitch = pano.Stitcher(calib_files)
        self.pano_stitch_init = False
        self.pano_stitch = None
        try:
            mat = pano_stitch.build(self.scale_factor, multi_band_blend)
            self.pano_stitch_init = True
            self.pano_stitch = pano_stitch
            return mat
        except:
            raise RuntimeError(f"Error: Failed to calibrate the input images {calib_files}")

    def build_pano(self, img_paths, multi_band_blend):
        if not self.pano_stitch: return None
        mat = self.pano_stitch.build_from_new_images(img_paths, multi_band_blend)
        return mat

    def write_img(self, path, mat):
        if not self.pano_stitch: return None
        pano.write_img(path, mat)

    def pano_stitch_single_camera(self, img_paths: list, out_filename=None, calib_files=None, return_img=False,
                                  multi_band_blend=0):
        """

        :param img_paths: The list of the image paths to stitched
        :param out_filename: The output filename
        :param calib_files: Camera calibration image file if exists
        :param multi_band_blend: -1 for merge blending, 0 for linear blending, k>0 for multiband blending.
                                Merge blending result cannot be saved directly a file.
        :param return_img: if True the resulting stitched image will be returned.
                            The result image would be a numpy array
        :return: A numpy array of the stitched image
        """

        if calib_files:
            try:
                self.init_pano_stitcher(calib_files, multi_band_blend)
            except:
                # raise RuntimeError(f"Error: Failed to calibrate the input images {calib_files}")
                if self.pano_stitch_init:
                    self.init_pano_stitcher(img_paths, multi_band_blend)

        if not self.pano_stitch_init:
            print("Please, initialize the pano object by provide the calibration files.")
            return None

        mat = self.build_pano(img_paths, multi_band_blend)

        if out_filename and multi_band_blend > 0:
            pano.write_img(out_filename, mat)

        if return_img:
            p = np.array(mat, copy=False)
            p[p < 0] = 0  # Replace negative values with zeros
            p[p > 1] = 1  # Replace negative values with zeros
            h, w, c = p.shape
            X = np.zeros((1, h, w, abs(multi_band_blend) * 3))

            X[0, :, :, :c] = p
            print("********++++==> ", X.shape, " ~ ", p.dtype, ", multi_band_blend: ", multi_band_blend)
            return X

    def pano_stitch_multi_camera(self, img_pattern, nb_cameras, nb_images, out_dir=None,
                                 multi_band_blend=0, return_img=False):
        """
            {
                "img_pattern": "data_stitching/Terrace/Input/{:05d}/{:05d}.jpg",
                "out_dir": "data_stitching/Terrace/Out2",
                "nb_camera": 14,
                "nb_images": 430,
            },

        :param img_pattern: Image pattern to build all camera image paths. It should contains an interger to index
                            the camera and the image of the camera (Mij).
                            ex: DIR/Input/{:05d}/{:05d}.jpg"
        :param nb_cameras: total number of camera
        :param nb_images: total number of images per camera to stitched
        :param out_dir: Output directory to save the output result. The
        :param multi_band_blend: -1 for merge blending, 0 for linear blending, k>0 for multi-band blending.
                                Merge blending result cannot be saved directly a file.
        :param return_img: if True the resulting stitched image will be returned.
                            The result image would be a numpy array
        :return: Return a list of numpy array of images. list[numpy]
        """

        for img_id in range(nb_images):

            file_list = [img_pattern.format(i, img_id) for i in range(nb_cameras)]

            print(file_list)
            try:
                self.init_pano_stitcher(file_list, multi_band_blend)
                break
            except:
                print(f"Error: Cannot stitch image [{img_id}]")

        if not self.pano_stitch_init:
            raise RuntimeError("Failed to find the projection parameters. Please add calibration "
                               "images in the same director as the images")

        result_list = []
        for img_id in range(total_img):

            file_list = [img_pattern.format(i, img_id) for i in range(nb_cameras)]

            mat = self.build_pano(file_list, multi_band_blend)
            print(f"Stitched image number: {img_id}/{total_img}")

            if return_img:
                result_list.append(np.array(mat, copy=False))

            if not out_dir and multi_band_blend > 0:
                output_path = f"{out_dir}/{img_id:05d}.jpg"
                pano.write_img(output_path, mat)

        if result_list:
            return result_list


if __name__ == "__main__":
    # prepare_data()

    import libpyopenpano as pano

    # help(pano)
    # Test Stitching
    pano.print_config()
    pano.init_config(cfg.pano_config_file)
    pano.print_config()

    mdata = [
        {
            "img_pattern": "/media/sf_Data/data_stitching/Terrace/Input/{:05d}/{:05d}.jpg",
            "out_dir": "/media/sf_Data/data_stitching/Terrace/Out2",
            "nb_cameras": 14,
            "total_img": 430,
        },
        {
            "img_pattern": "/media/sf_Data/data_stitching/Terrace/Input/{:05d}/{:05d}.jpg",
            "out_dir": "/media/sf_Data/data_stitching/Terrace/Out2",
            "nb_cameras": 14,
            "total_img": 430,
        }
    ]

    id = 0
    # img_dir = "/media/sf_Data/data_stitching/Terrace/Input/{:05d}/{:05d}.jpg"
    out_dir = mdata[id]["out_dir"]
    nb_camera = mdata[id]["nb_cameras"]
    total_img = mdata[id]["total_img"]
    os.makedirs(out_dir, exist_ok=True)
    output_result = None

    nb_stitched_img = 0
    stitcher = None
    for img_id in range(total_img):
        print(f"-----------------------------{img_id}------------------------")
        file_list = [mdata[id]["img_pattern"].format(i, img_id) for i in range(mdata[id]["nb_cameras"])]
        output_result = f"{out_dir}/{img_id:05d}.jpg"

        print(file_list)
        print(output_result)

        stitcher = None
        try:
            stitcher = pano.Stitcher(file_list)
            mat = stitcher.build()
            pano.write_img(output_result, mat)
            break
        except:
            print(f"Error: Cannot stitch image [{img_id}] - [{output_result}]")

    print(f"First image stitched: {stitcher} --> Location: {output_result}")
    multi_band_blend = 0  # 0 is for linear blending
    time.sleep(10)
    for img_id in range(total_img):
        print(f"-----------------------------{img_id}------------------------")
        file_list = [mdata[id]["img_pattern"].format(i, img_id) for i in range(mdata[id]["nb_cameras"])]
        output_result = f"{out_dir}/{img_id:05d}.jpg"

        try:
            print(f"Try to build from new images {img_id}/{total_img}")
            # print(file_list)
            mat = stitcher.build_from_new_images(file_list, multi_band_blend)
            print(f"Done building from new images {img_id}/{total_img}")
            # time.sleep(10)
            pano.write_img(output_result, mat)
        except:
            print(f"[build_from_new_images] Error: Cannot stitch image [{img_id}] - [{output_result}]")

    # pano.test_extrema(mat, 1)
    print("done stitching!", nb_stitched_img)
    # pano.print_config()

    # p = np.array(mat, copy=False)
    # plt.imshow(p)
    # plt.show()
