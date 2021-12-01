import sys

import model_stitching
import argparse
# import tensorflow as tf
import os
import glob
import numpy as np
import time

import panowrapper as pw

import math
import cv2
import project_settings as cfg
import time


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def stitch(files, model_type, outdir, scale_factor, compare_result=True):

    global panow
    global model
    img_merge = panow.pano_stitch_single_camera(files, multi_band_blend=-5, return_img=True)
    # panow.print_config()
    if img_merge is None:
        print(f"failed to stitch the images {files}")
        sys.exit()

    img_mbb = None
    if compare_result:
        img_mbb = panow.pano_stitch_single_camera(files, multi_band_blend=20, return_img=True)
        img_mbb = img_mbb.astype(np.float32) * 255.
        img_mbb = np.clip(img_mbb, 0, 255).astype('uint8')
        img_mbb = img_mbb[0, :, :, :]
        print(f"=> Shape im_merge: {img_merge.shape}, shape of img_mbb: {img_mbb.shape}")

    if model_type == "ddis":
        # Pad image with zeros to the nearest power of 2
        h = nearest_mult_n(img_merge.shape[1]) - img_merge.shape[1]
        w = nearest_mult_n(img_merge.shape[2]) - img_merge.shape[2]
        img_merge = np.pad(img_merge, ((0, 0), (0, h), (0, w), (0, 0)), mode='constant')
        if compare_result:
            img_mbb = np.pad(img_mbb, ((0, h), (0, w), (0, 0)), mode='constant')
            print(f"=> New Shape im_merge: {img_merge.shape}, shape of img_mbb: {img_mbb.shape}")

    start_time = time.time()
    result = model.simple_stitch(img_merge, out_dir=outdir, scale_factor=scale_factor,
                                 suffix=model_type, return_image=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    if compare_result:
        m_pnsr = calculate_psnr(img_mbb, result)
        m_ssim = calculate_ssim(img_mbb, result)
        print(f"File {files[0]}: PNSR: {m_pnsr}, and SSIM: {m_ssim}")


parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
parser.add_argument("imgpath", type=str, nargs="*", help="Path to input image")
parser.add_argument("--imgdir", type=str, default=None, help="Image directory")
parser.add_argument("--outdir", type=str, default=None, help="Output Result directory")
parser.add_argument("--calib_dir", type=str, default=None, help="Camera Calibration directory that contains images")
parser.add_argument("--calib_pattern", type=str, default=None,
                    help="Calibration images pattern. Should contains the camera id and image id."
                         "Ex: 'DIR/Input/{camID:05d}/{imgID:05d}.jpg'")
parser.add_argument("--input_pattern", type=str, default=None,
                    help="Input image pattern contain the string to retrieved images. Should contains the camera id and image id."
                         "Ex: 'DIR/Input/{camID:05d}/{imgID:05d}.jpg'")
parser.add_argument("-nbc", "--nb_cameras", type=int, default=0, help="Total number of cameras")
parser.add_argument("-nbi", "--nb_images", type=int, default=0, help="Maximun number of images to use for calibration")
parser.add_argument("-nbis", "--nb_stitch_images", type=int, default=0, help="Maximun number of images to stitch in the pattern")
parser.add_argument('--compare_result', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument("--scale_factor", type=float, default=1.0,
                    help="Input image scale factor [to be divide]."
                         "For example use 3 to scale the input images by a factor of 1/3")

parser.add_argument("--model", type=str, default="ddis",
                    help="Use either image super resolution (is), "
                         "expanded super resolution (eis), "
                         "denoising auto encoder img stitching (dis), "
                         "deep denoising img stitching (ddis) or res net sr (rnis)")


def nearest_power_2(x: int):
    return 1 << (x-1).bit_length()


def nearest_mult_n(m, n=4):
    return m if m % n == 0 else ((m//n)+1)*n


args = parser.parse_args()
print(args)
model_type = str(args.model).lower()
if not model_type in ["is", "eis", "dis", "ddis", "rnis", "distilled_rnis"]:
    raise ValueError('Model type must be either "is", "eis", "dis", '
                     '"ddis", "rnis" or "distilled_rnis"')

panow = None
model = None

if __name__ == "__main__":
    # path = args.imgpath
    # print("List of files: ", args.imgpath)
    # print("Images directory: ", args.imgdir)
    if not args.imgpath and args.imgdir is None and args.input_pattern:
        ValueError("Please provide the list of files or the directory or the pattern containing the images to be stitched")

    # for p in path:
    files = args.imgpath
    if not args.imgpath and args.imgdir is not None:
        files = []
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.imgdir, ext)))

    print("List of files: ", files)
    if model_type == "is":  # Work
        model = model_stitching.ImageStitchingModel()
    elif model_type == "eis":  # Work
        model = model_stitching.ExpantionStitching()
    elif model_type == "dis":  # Do Not Work
        model = model_stitching.DenoisingAutoEncoderStitch()
    elif model_type == "ddis":  # Do Not Work
        model = model_stitching.DeepDenoiseStitch()
    elif model_type == "rnis":  # Work
        model = model_stitching.ResNetStitch()
    elif model_type == "distilled_rnis":  # Not Trained Yet
        # model = model_stitching.DistilledResNetStitch()
        print("The Distilled model has not been trained yet.")
        sys.exit()
    else:
        model = model_stitching.ImageStitchingModel()

    panow = pw.PanoWrapper(scale_factor=args.scale_factor)

    # initialize pano stitch
    if args.calib_pattern is not None and args.nb_cameras and args.nb_images:
        panow.init_pano_from_dir(args.calib_pattern, args.nb_cameras, args.nb_images)

    if not panow.is_pano_initialize() and args.calib_dir is not None:
        panow.init_pano_stitcher(args.calib_dir, multi_band_blend=0)

    if not panow.is_pano_initialize():
        panow.init_pano_stitcher(files, multi_band_blend=0)

    if not panow.is_pano_initialize():
        print("Failed to initialized pano object")
        sys.exit()

    m_outdir = f"{cfg.dataset_folder}/out_result" if args.outdir is None else args.outdir
    # if args.input_pattern is not None and args.nb_stitch_images:
    #     for img_id in range(args.nb_stitch_images):

    #         file_list = [args.input_pattern.format(camID=i, imgID=img_id) for i in range(args.nb_cameras)]
            
    #         img_merge = panow.pano_stitch_single_camera(file_list, multi_band_blend=-5, return_img=True)

    #         if img_merge is None:
    #             print(f"failed to stitch the images {files}")

    # stitch(files, model_type=model_type, outdir=m_outdir, compare_result=args.compare_result, scale_factor=args.scale_factor)


    if args.input_pattern is not None and args.nb_stitch_images:
        for img_id in range(args.nb_stitch_images):
            
            file_list = [args.input_pattern.format(camID=i, imgID=img_id) for i in range(args.nb_cameras)]
            stitch(file_list, model_type=model_type, outdir=m_outdir, compare_result=args.compare_result, scale_factor=args.scale_factor)

    else:
        stitch(files, model_type=model_type, outdir=m_outdir, compare_result=args.compare_result, scale_factor=args.scale_factor)

    # # img_merge = panow.pano_stitch_single_camera(files, calib_files=files, multi_band_blend=-5, return_img=True)
    # img_merge = panow.pano_stitch_single_camera(files, calib_files=None, multi_band_blend=-5, return_img=True)
    # # panow.print_config()
    # if img_merge is None:
    #     print(f"failed to stitch the images {files}")
    #     sys.exit()

    # img_mbb = None
    # if args.compare_result:
    #     img_mbb = panow.pano_stitch_single_camera(files, calib_files=None, multi_band_blend=20, return_img=True)
    #     img_mbb = img_mbb.astype(np.float32) * 255.
    #     img_mbb = np.clip(img_mbb, 0, 255).astype('uint8')
    #     img_mbb = img_mbb[0, :, :, :]
    #     print(f"=> Shape im_merge: {img_merge.shape}, shape of img_mbb: {img_mbb.shape}")

    # if model_type == "ddis":
    #     # Pad image with zeros to the nearest power of 2
    #     h = nearest_mult_n(img_merge.shape[1]) - img_merge.shape[1]
    #     w = nearest_mult_n(img_merge.shape[2]) - img_merge.shape[2]
    #     img_merge = np.pad(img_merge, ((0, 0), (0, h), (0, w), (0, 0)), mode='constant')
    #     if args.compare_result:
    #         img_mbb = np.pad(img_mbb, ((0, h), (0, w), (0, 0)), mode='constant')
    #         print(f"=> New Shape im_merge: {img_merge.shape}, shape of img_mbb: {img_mbb.shape}")

    # # print(img_merge.shape)
    # # model.stitch(files, scale_factor=args.scale_factor, suffix=model_type)
    # outdir = f"{cfg.dataset_folder}/out_result" if args.outdir is None else args.outdir

    # start_time = time.time()
    # result = model.simple_stitch(img_merge, out_dir=outdir, scale_factor=args.scale_factor,
    #                              suffix=model_type, return_image=True)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # if args.compare_result:
    #     m_pnsr = calculate_psnr(img_mbb, result)
    #     m_ssim = calculate_ssim(img_mbb, result)
    #     print(f"File {files[0]}: PNSR: {m_pnsr}, and SSIM: {m_ssim}")
