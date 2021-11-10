import model_stitching
import argparse
# import tensorflow as tf
import os
import glob
import time

import panowrapper as pw

parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
parser.add_argument("imgpath", type=str, nargs="*", help="Path to input image")
parser.add_argument("--imgdir", type=str, default=None, help="Image directory")
parser.add_argument("--outdir", type=str, default=None, help="Output Result directory")
parser.add_argument("--scale_factor", type=float, default=1.0,
                    help="Input image scale factor [to be divide]."
                         "For example use 3 to scale the input images by a factor of 1/3")

parser.add_argument("--model", type=str, default="is",
                    help="Use either image super resolution (is), "
                         "expanded super resolution (eis), "
                         "denoising auto encoder sr (dis), "
                         "deep denoising sr (ddis) or res net sr (rnis)")


args = parser.parse_args()

model_type = str(args.model).lower()
if not model_type in ["is", "eis", "dis", "ddis", "rnis", "distilled_rnis"]:
    raise ValueError('Model type must be either "is", "eis", "dis", '
                     '"ddis", "rnis" or "distilled_rnis"')


if __name__ == "__main__":
    # path = args.imgpath
    print("List of files: ", args.imgpath)
    print("Images directory: ", args.imgdir)
    if not args.imgpath and args.imgdir is None:
        ValueError("Please provide the list of files or the directory containing the images to be stitched")

    # for p in path:
    files = args.imgpath
    if not args.imgpath and args.imgdir is not None:
        files = []
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.imgdir, ext)))

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
        model = model_stitching.DistilledResNetStitch()
    else:
        model = model_stitching.ImageStitchingModel()

    panow = pw.PanoWrapper()
    img_merge = panow.pano_stitch_single_camera(files, multi_band_blend=-1, return_img=True)
    # panow.print_config()

    # model.stitch(files, scale_factor=args.scale_factor, suffix=model_type)
    outdir = "/media/sf_Data/data_stitching/deepstitch-dataset/out_result"
    model.simple_stitch(img_merge, out_dir=outdir, scale_factor=args.scale_factor, suffix=model_type)
