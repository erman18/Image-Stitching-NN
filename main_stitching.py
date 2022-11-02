import sys

import model_stitching
import SSL_models
import argparse
# import tensorflow as tf
import os
import glob
import numpy as np
import time

import panowrapper as pw

import math
import cv2
import constant as cfg
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

CST_SEAM = "seam" # Constant for seam-based
CST_MULB = "multiband" # Constant for multiband blending
CST_DEEP = "ours" # Constant for deep learning approach
from scipy.linalg import sqrtm
class FIDMetric:
    def __init__(self, model=None) -> None:
        self.model = model
        self.__act1 = {
            CST_SEAM: [],
            CST_MULB: [],
            CST_DEEP: [],
        }
        self.__act2 = {
            CST_SEAM: [],
            CST_MULB: [],
            CST_DEEP: [],
        }

    def add_image_seam(self, images1, images2):
        self.__warped_add(images1, images2, CST_SEAM)

    def add_image_mbb(self, images1, images2):
        self.__warped_add(images1, images2, CST_MULB)

    def add_image_deep(self, images1, images2):
        self.__warped_add(images1, images2, CST_DEEP)

    def __warped_add(self, image1, image2, method):

        if image1.shape[-1] > 3:
            nb_layers = int(image1.shape[-1]//3)

            # Save SandFall layers
            for i in range(nb_layers):
                img = image1[:, :, :, i*3:(i+1)*3]
                mask = img != 0.0
                masked_img = mask * image2
                self.__add_image(img, masked_img, method)
        else:
            self.__add_image(image1, image2, method)

    # calculate frechet inception distance
    def __add_image(self, images1, images2, method):

        if images1.ndim == 3:
            images1 = np.expand_dims(images1, 0)
        if images2.ndim == 3:
            images2 = np.expand_dims(images2, 0)

        # Assume image values are between 0 and 255.
        processed1 = images1 / 127.5 - 1.0
        processed2 = images2 / 127.5 - 1.0
        # calculate activations
        act1 = self.model.predict(processed1)
        act2 = self.model.predict(processed2)
        self.__act1[method].append(act1.ravel())
        self.__act2[method].append(act2.ravel())
    
    # calculate frechet inception distance
    def __calculate_fid(self, method):
        if not (self.__act1[method] and self.__act1[method]):
            return 0
        act1 = np.array(self.__act1[method], dtype=np.float32)
        act2 = np.array(self.__act2[method], dtype=np.float32)
        # print("-------------------------------->", act2.shape)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # print(sigma1, sigma2.ndim)
        # calculate sqrt of product between cov
        sig = sigma1.dot(sigma2)
        if sig.ndim > 0:
            covmean = sqrtm(sig)
        else:
            covmean = np.sqrt(sig)
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        if sig.ndim > 0:
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        else:
            fid = ssdiff + sigma1 + sigma2 - 2.0 * covmean
        return fid
    
    def getvalues(self):
        # return self.__calculate_fid()
        return {
            CST_SEAM: self.__calculate_fid(CST_SEAM),
            CST_MULB: self.__calculate_fid(CST_MULB),
            CST_DEEP: self.__calculate_fid(CST_DEEP),
        }
    
    def reset(self, method):
        self.__act1[method] = []
        self.__act2[method] = []

    def reset_all(self):
        self.reset(CST_SEAM)
        self.reset(CST_MULB)
        self.reset(CST_DEEP)

class ISScore:
    def __init__(self, model=None) -> None:
        self.model = model
        self.__p_yx = {
            CST_SEAM: [],
            CST_MULB: [],
            CST_DEEP: [],
        }
    
    def add_image_seam(self, image):
        self.__add_image(image, CST_SEAM)

    def add_image_mbb(self, image):
        self.__add_image(image, CST_MULB)

    def add_image_deep(self, image):
        self.__add_image(image, CST_DEEP)

    # calculate the inception score for p(y|x)
    def __add_image(self, image, method):

        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        # assume image values are between  0-255
        processed = image.astype(np.float32) / 127.5 - 1.0
        # calculate p(y|x)
        p_yx = self.model.predict(processed)
        
        self.__p_yx[method].append(p_yx.ravel())
    
    # calculate the inception score for p(y|x)
    def __calculate_inception_score(self, method, eps=1E-16):
        if not (self.__p_yx[method]):
            return 0
        # calculate p(y|x)
        p_yx = np.array(self.__p_yx[method], dtype=np.float32)
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # kl divergence for each image
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the logs
        is_score = np.exp(avg_kl_d)
        return is_score
    
    def getvalues(self):
        return {
            CST_SEAM: self.__calculate_inception_score(CST_SEAM),
            CST_MULB: self.__calculate_inception_score(CST_MULB),
            CST_DEEP: self.__calculate_inception_score(CST_DEEP),
        }
    
    def reset(self, method):
        self.__p_yx[method] = []

    def reset_all(self):
        self.reset(CST_SEAM)
        self.reset(CST_MULB)
        self.reset(CST_DEEP)

class SGScore:
    def __init__(self, model=None) -> None:
        self.model = model
        self.__p_yx = {
            CST_SEAM: [],
            CST_MULB: [],
            CST_DEEP: [],
        }
    
    def add_image_seam(self, image):
        self.__add_image(image, CST_SEAM)

    def add_image_mbb(self, image):
        self.__add_image(image, CST_MULB)

    def add_image_deep(self, image):
        self.__add_image(image, CST_DEEP)

    # calculate the inception score for p(y|x)
    def __add_image(self, image, method):

        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        # assume image values are between  0-255
        processed = image.astype(np.float32) / 127.5 - 1.0
        # calculate p(y|x)
        p_yx = self.model.predict(processed)
        
        self.__p_yx[method].append(p_yx.ravel())
    
    # calculate the inception score for p(y|x)
    def __calculate_inception_score(self, method, eps=1E-16):
        if not (self.__p_yx[method]):
            return 0
        # calculate p(y|x)
        p_yx = np.array(self.__p_yx[method], dtype=np.float32)
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # kl divergence for each image
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the logs
        sg_score = avg_kl_d # np.exp(avg_kl_d)
        return sg_score
    
    def getvalues(self):
        return {
            CST_SEAM: self.__calculate_inception_score(CST_SEAM),
            CST_MULB: self.__calculate_inception_score(CST_MULB),
            CST_DEEP: self.__calculate_inception_score(CST_DEEP),
        }
    
    def reset(self, method):
        self.__p_yx[method] = []

    def reset_all(self):
        self.reset(CST_SEAM)
        self.reset(CST_MULB)
        self.reset(CST_DEEP)

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

class LPIPSMetric:
    """ Learned Perceptual Image Patch Similarity (LPIPS) between img1 and img2 """

    def __init__(self,
                pb_fname="/home/UFAD/enghonda/.lpips/net_vgg_v0.1.pb"):
        loaded = tf.saved_model.load(pb_fname)
        self.frozen_func = loaded.signatures['serving_default']
        self.__distance = {
            CST_SEAM: [],
            CST_MULB: [],
            CST_DEEP: [],
        }

    def add_image_seam(self, images1, images2):
        self.__warped_add(images1, images2, CST_SEAM)

    def add_image_mbb(self, images1, images2):
        self.__warped_add(images1, images2, CST_MULB)

    def add_image_deep(self, images1, images2):
        self.__warped_add(images1, images2, CST_DEEP)

    def __warped_add(self, image1, image2, method):

        if image1.shape[-1] > 3:
            nb_layers = int(image1.shape[-1]//3)

            # Save SandFall layers
            for i in range(nb_layers):
                img = image1[:, :, :, i*3:(i+1)*3]
                mask = img != 0.0
                masked_img = mask * image2
                self.__add_image(img, masked_img, method)
        else:
            self.__add_image(image1, image2, method)

    def __add_image(self, y_true, y_pred, method):

        if y_true.ndim == 3:
            y_true = np.expand_dims(y_true, 0)
        if y_pred.ndim == 3:
            y_pred = np.expand_dims(y_pred, 0)
            
        # calculate activations
        input0 = np.transpose(y_true, [0, 3, 1, 2]) / 127.5 - 1.0
        input1 = np.transpose(y_pred, [0, 3, 1, 2]) / 127.5 - 1.0
        distance = self.frozen_func(
            in0=tf.convert_to_tensor(input0, dtype=tf.float32), 
            in1=tf.convert_to_tensor(input1, dtype=tf.float32)
        )
        
        self.__distance[method].append(distance['185'].numpy())
    
    def __getdiscance(self, method):        
        return np.array(self.__distance[method], 
                dtype=np.float32).mean() if self.__distance[method] else 0.0
    
    def getvalues(self):
        return {
            CST_SEAM: self.__getdiscance(CST_SEAM),
            CST_MULB: self.__getdiscance(CST_MULB),
            CST_DEEP: self.__getdiscance(CST_DEEP),
        }

    def reset(self, method):
        self.__distance[method] = []

    def reset_all(self):
        self.reset(CST_SEAM)
        self.reset(CST_MULB)
        self.reset(CST_DEEP)
        
fid_metric = FIDMetric()
is_metric = ISScore()
sg_metric = SGScore()
lpips_metric = LPIPSMetric()

def save_blending_result(img_merge, outdir, file_prefix, nb_layers=None):
    
    img_shape = img_merge.shape
    if len(img_shape) < 4:
        cfg.PRINT_ERROR("Invalid Image Shape")
        exit(1)

    if not nb_layers:
        nb_layers = int(img_shape[3]//3)

    # Save SandFall layers
    for i in range(nb_layers):
        filename = os.path.join(outdir, file_prefix + str(i) + ".jpg")
        img = img_merge[0, :,:, i*3:(i+1)*3]
        cv2.imwrite(filename, cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

def stitch(files, model_type, outdir, scale_factor, compare_result=True):

    global panow
    global model
    os.makedirs(m_outdir, exist_ok=True)
    suffix = time.strftime("_%Y%m%d-%H%M%S")
    img_merge = panow.pano_stitch_single_camera(files, multi_band_blend=-1, return_img=True)
    save_blending_result(img_merge, outdir, "merge_layer", min(abs(cfg.sandfall_layer), img_merge.shape[-1]//3))
    sandfall_block = panow.pano_stitch_single_camera(files, multi_band_blend=-5, return_img=True)
    # panow.print_config()
    if sandfall_block is None:
        print(f"failed to stitch the images {files}")
        sys.exit()

    # Save SandFall layers
    save_blending_result(sandfall_block, outdir, "sandfall_layer", abs(cfg.sandfall_layer))
    # for i in range(abs(cfg.sandfall_layer)):
    #     filename = os.path.join(outdir, "sandfall_layer" + str(i) + ".jpg")
    #     img = sandfall_block[0, :,:, i*3:(i+1)*3]
    #     cv2.imwrite(filename, cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    img_mbb = None
    if compare_result:
        out_filename=os.path.join(outdir, "multiband_20_" + suffix + ".jpg")
        img_mbb = panow.pano_stitch_single_camera(files, out_filename=out_filename, multi_band_blend=20, return_img=True)
        img_mbb = img_mbb.astype(np.float32) * 255.
        img_mbb = np.clip(img_mbb, 0, 255).astype('uint8')
        img_mbb = img_mbb[0, :, :, :]

        out_seam_img=os.path.join(outdir, "seam_blending_" + suffix + ".jpg")
        img_seam = panow.pano_stitch_single_camera(files, out_filename=out_seam_img, multi_band_blend=-25, return_img=True)
        img_seam = img_seam.astype(np.float32) * 255.
        img_seam = np.clip(img_seam, 0, 255).astype('uint8')
        img_seam = img_seam[0, :, :, :]

        print(f"=> Shape im_merge: {sandfall_block.shape}, shape of img_mbb: {img_mbb.shape}, shape of img_seam: {img_seam.shape}")

    if model_type == "ddis" or model_type == "unddis":
        # Pad image with zeros to the nearest power of 2
        h = nearest_mult_n(sandfall_block.shape[1]) - sandfall_block.shape[1]
        w = nearest_mult_n(sandfall_block.shape[2]) - sandfall_block.shape[2]
        sandfall_block = np.pad(sandfall_block, ((0, 0), (0, h), (0, w), (0, 0)), mode='constant')
        if compare_result:
            img_mbb = np.pad(img_mbb, ((0, h), (0, w), (0, 0)), mode='constant')
            print(f"=> New Shape im_merge: {sandfall_block.shape}, shape of img_mbb: {img_mbb.shape}")

    start_time = time.time()
    result = model.simple_stitch(sandfall_block, out_dir=outdir, scale_factor=scale_factor,
                                 suffix=model_type + str(suffix), return_image=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    if compare_result:
        m_pnsr = calculate_psnr(img_mbb, result)
        m_ssim = calculate_ssim(img_mbb, result)
        cfg.PRINT_INFO(f"Computing Metrics - im_mg.shape: {img_merge.shape}, result.shape: {result.shape}")
        inceptionv3_model = InceptionV3(include_top=False, pooling='avg', input_shape=result.shape)
        fid_metric.model = inceptionv3_model
        is_metric.model = inceptionv3_model
        sg_metric.model = inceptionv3_model
        
        is_metric.add_image_deep(result)
        is_metric.add_image_mbb(img_mbb)
        is_metric.add_image_seam(img_seam)

        sg_metric.add_image_deep(result)
        sg_metric.add_image_mbb(img_mbb)
        sg_metric.add_image_seam(img_seam)

        fid_metric.add_image_deep(img_merge, result)
        fid_metric.add_image_mbb(img_merge, img_mbb)
        fid_metric.add_image_seam(img_merge, img_seam)

        lpips_metric.add_image_mbb(img_merge, result)
        lpips_metric.add_image_deep(img_merge, img_mbb)
        lpips_metric.add_image_seam(img_merge, img_seam)
        print(f"PNSR: {m_pnsr}, and SSIM: {m_ssim}, Output Dir: {outdir}")


parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
# parser.add_argument("imgpath", type=str, nargs="*", help="Path to input image")
parser.add_argument("--imgdir", nargs='?', type=str, default=None, help="Image directory")
parser.add_argument("--outdir", type=str, default=None, help="Output Result directory")
parser.add_argument("--calib_dir", nargs='?', type=str, default=None, help="Camera Calibration directory that contains images")
parser.add_argument("--calib_pattern", nargs='?', type=str, default=None,
                    help="Calibration images pattern. Should contains the camera id and image id."
                         "Ex: 'DIR/Input/{camID:05d}/{imgID:05d}.jpg'")
parser.add_argument("--input_pattern", nargs='?', type=str, default=None,
                    help="Input image pattern contain the string to retrieved images. Should contains the camera id and image id."
                         "Ex: 'DIR/Input/{camID:05d}/{imgID:05d}.jpg'")
parser.add_argument("-nbc", "--nb_cameras", nargs='?', type=int, default=0, help="Total number of cameras")
parser.add_argument("-nbi", "--nb_images", nargs='?', type=int, default=0, help="Maximun number of images to use for calibration")
parser.add_argument("-nbis", "--nb_stitch_images", nargs='?', type=int, default=0, help="Maximun number of images to stitch in the pattern")
parser.add_argument('--compare_result', nargs='?', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument("--scale_factor", nargs='?', type=float, default=1.0,
                    help="Input image scale factor [to be divide]."
                         "For example use 3 to scale the input images by a factor of 1/3")

parser.add_argument('--files', nargs='+', help='List of image file to stitch', metavar='file1.jpg file2.jpg file3.png')
parser.add_argument("--dfs", nargs='?', type=str, default="MCMI", choices=["MCMI", "SCMI", "MCSI", "LIST", "IDIR"],
                    help="Dataset File Structure (DFS) \n"
                         "MCMI: Multi-Camera Multi-Image (indicate both the 'camID' and the 'imgID' in the file pattern \n"
                         "SCMI: Single-Camera Multi-Image ('imgID' in the file pattern"
                         "MCSI: Multi-Camera Multi-Image ('camID' in the file pattern"
                         "LIST: list of files image to stitch and provide the list in the --files param"
                         "IDIR: Image Directory of files with *.jpg, *.jpeg, *.png, and *.bmp will be retrieve in the folder indicate by --imgdir"
                         )
parser.add_argument("--metric", type=str, default=None, 
                    help="Pretrained Metric Weight to use: "
                    "lpips: Learned Perceptual Image Patch Similarity"
                    "is: Inception Score (IS)"
                    "pnsr: Peak Signal-to-Noise Ratio"
                    "ssim: SSIM - Structural Similarity Index Measure "
                    )

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
if not model_type in ["is", "eis", "dis", "ddis", "rnis", "distilled_rnis", "unrnis", "unddis"]:
    raise ValueError('Model type must be either "is", "eis", "dis", '
                     '"ddis", "rnis" or "distilled_rnis", "unrnis"')

panow = None
model = None

if __name__ == "__main__":

    if args.imgdir is None or args.input_pattern is None:
        ValueError("Please provide the list of files or the directory or the pattern containing the images to be stitched")
    
    files = []
    if args.dfs == "MCMI":
        files = [[args.input_pattern.format(camID=i, imgID=img_id) for i in range(args.nb_cameras)] for img_id in range(args.nb_stitch_images)]
    elif args.dfs == "SCMI":
        files = [args.input_pattern.format(imgID=img_id) for img_id in range(args.nb_stitch_images)]
    elif args.dfs == "MCSI":
        files = [args.input_pattern.format(camID=i) for i in range(args.nb_cameras)]
    elif args.dfs == "LIST":
        files = args.files
    elif args.dfs == "IDIR":
        import re
        # exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        exts = ["jpg", "jpeg", "png", "bmp"]
        for ext in exts:
            files.extend([os.path.join(args.imgdir, filename) 
            for filename in os.listdir(args.imgdir) if re.search(r'\.' + ext + '$', filename, re.IGNORECASE)])
            # files.extend(glob.glob(os.path.join(args.imgdir, ext)))
        print(files)
    else:
        ValueError("Please indicate the dataset file structure for your images")

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
    elif model_type == "unrnis":  # Work
        model = SSL_models.ResNetStitch(metric=args.metric)
    elif model_type == "unddis":  # Work
        model = SSL_models.DeepDenoiseStitch(metric=args.metric)
    elif model_type == "distilled_rnis":  # Not Trained Yet
        # model = model_stitching.DistilledResNetStitch()
        print("The Distilled model has not been trained yet.")
        sys.exit()
    else:
        model = model_stitching.ImageStitchingModel()

    panow = pw.PanoWrapper(scale_factor_x=args.scale_factor, scale_factor_y=args.scale_factor)

    ## Check for image parameters
    # initialize pano stitch
    if args.calib_pattern and args.nb_cameras and args.nb_images:
        panow.init_pano_from_dir(args.calib_pattern, args.nb_cameras, args.nb_images)

    elif not panow.is_pano_initialize() and args.calib_dir is not None:
        panow.init_pano_stitcher(args.calib_dir, multi_band_blend=0)

    elif not panow.is_pano_initialize():
        panow.init_pano_stitcher(files, multi_band_blend=0)

    if not panow.is_pano_initialize():
        cfg.PRINT_ERROR("Failed to initialized pano object")
        sys.exit()

    m_outdir = f"{cfg.dataset_folder}/out_result" if args.outdir is None else args.outdir

    if args.dfs == "MCMI":
        for idx, file_list in enumerate(files):

            # file_list = [args.input_pattern.format(camID=i, imgID=img_id) for i in range(args.nb_cameras)]
            print(f"==> Processing Image {idx+1}/{len(files)}")
            stitch(file_list, model_type=model_type, outdir=m_outdir, compare_result=args.compare_result, scale_factor=args.scale_factor)

    else:
        stitch(files, model_type=model_type, outdir=m_outdir, compare_result=args.compare_result, scale_factor=args.scale_factor)

    if args.compare_result:
        iss = is_metric.getvalues()
        sgs = sg_metric.getvalues()
        fid = fid_metric.getvalues()
        lpips_distance = lpips_metric.getvalues()
        print(f"iss: {iss}, sgs: {sgs}, fid: {fid}, lpips_distance: {lpips_distance}")

