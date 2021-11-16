from __future__ import print_function, division

import argparse

from keras.utils.vis_utils import plot_model
import model_stitching
import img_utils
import project_settings as cfg

from model_stitching import DeepDenoiseStitch as DDStitch
from model_stitching import DistilledResNetStitch as DResStitch
from model_stitching import ResNetStitch as ResNetStitch
from model_stitching import ImageStitchingModel as DPImgStitch
from model_stitching import ExpantionStitching as ExpStitch
from model_stitching import DenoisingAutoEncoderStitch as DAutoEncoderStitch

model_directory = {'DDStitch': DDStitch,
                   'DResStitch': DResStitch,
                   'ResNetStitch': ResNetStitch,
                   'DPImgStitch': DPImgStitch,
                   'ExpStitch': ExpStitch,
                   'DAutoEncoderStitch': DAutoEncoderStitch,
                }

parser = argparse.ArgumentParser()
# parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--model', default="DDStitch", help="Deep Denoise Stitching Model", type=str)
parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument('--nb_epochs', default=10, type=int)
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

args = parser.parse_args()
print("==> Training Argument: ", args)
net = model_directory[args.model]
model_name = args.model


def train(height, width, nb_epochs=10, batch_size=32, save_arch=False, load_weights=True):

    # stitch_model = model_stitching.NonLocalResNetStitching()
    stitch_model = net() # model_stitching.DeepDenoiseStitch()
    stitch_model.create_model(height=height, width=width, load_weights=load_weights)

    if save_arch:
        plot_model(stitch_model.model, to_file=f"architectures/model_img/{model_name}.png", show_shapes=True,
                   show_layer_names=True)

    stitch_model.fit(nb_epochs=nb_epochs, batch_size=batch_size)


def save_model_plots():
    for modname in model_directory:
        print(f"=> Model: {modname}")
        network = model_directory[modname]
        stitch_model = network()
        stitch_model.create_model(height=128, width=128, load_weights=False)
        plot_model(stitch_model.model, to_file=f"architectures/model_img/{modname}.png", show_shapes=True,
                   show_layer_names=True)


if __name__ == "__main__":
    """
    Plot the models
    """
    train(height=cfg.patch_size, width=cfg.patch_size, save_arch=False, nb_epochs=args.nb_epochs,
          batch_size=32, load_weights=args.load_weights)
    # save_model_plots()