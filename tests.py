from __future__ import print_function, division

from keras.utils.vis_utils import plot_model
import models
import model_stitching
import img_utils
import project_settings as cfg

if __name__ == "__main__":
    path = r"headline_carspeed.jpg"
    val_path = "val_images/"

    scale = 2

    """
    Plot the models
    """

    # model = models.ImageSuperResolutionModel(scale).create_model()
    # plot_model(model, to_file="architectures/SRCNN.png", show_shapes=True, show_layer_names=True)

    # model = models.ExpantionSuperResolution(scale).create_model()
    # plot_model(model, to_file="architectures/ESRCNN.png", show_layer_names=True, show_shapes=True)

    # model = models.DenoisingAutoEncoderSR(scale).create_model()
    # plot_model(model, to_file="architectures/Denoise.png", show_layer_names=True, show_shapes=True)

    # model = models.DeepDenoiseSR(scale).create_model()
    # plot_model(model, to_file="architectures/Deep Denoise.png", show_layer_names=True, show_shapes=True)

    # model = models.ResNetSR(scale).create_model()
    # plot_model(model, to_file="architectures/ResNet.png", show_layer_names=True, show_shapes=True)

    # model = models.GANImageSuperResolutionModel(scale).create_model(mode='train')
    # plot_model(model, to_file='architectures/GAN Image SR.png', show_shapes=True, show_layer_names=True)

    # model = models.DistilledResNetSR(scale).create_model()
    # plot_model(model, to_file='architectures/distilled_resnet_sr.png', show_layer_names=True, show_shapes=True)

    # model = models.NonLocalResNetSR(scale).create_model()
    # plot_model(model, to_file='architectures/non_local_resnet_sr.png', show_layer_names=True, show_shapes=True)

    # model = model_stitching.NonLocalResNetSR().create_model()
    # plot_model(model, to_file="architectures/SRCNN_stitch.png", show_shapes=True, show_layer_names=True)

    """
    Train Super Resolution => Did work
    """

    # sr = models.ImageSuperResolutionModel(scale)
    # sr.create_model(height=64, width=64)
    # sr.fit(nb_epochs=250)

    """
    Train Stitching Model CNN
    """

    # sr = model_stitching.ImageStitchingModel()
    # sr.create_model(height=128, width=128)
    # sr.fit(nb_epochs=20, batch_size=32)

    """
    Train image stitching model Resnet
    """

    sr = model_stitching.ResNetStitch()
    sr.create_model(height=128, width=128)
    sr.fit(nb_epochs=10, batch_size=32)

    """
    Train image stitching model Resnet
    """

    # sr = model_stitching.ExpantionStitching()
    # sr.create_model(height=128, width=128)
    # sr.fit(nb_epochs=10, batch_size=32)

    """
    Train image stitching model Resnet
    """

    # sr = model_stitching.DenoisingAutoEncoderStitch()
    # sr.create_model(height=128, width=128)
    # sr.fit(nb_epochs=10, batch_size=32)

    """
    Train ExpantionSuperResolution => Did not work
    """

    # esr = models.ExpantionSuperResolution(scale)
    # esr.create_model()
    # esr.fit(nb_epochs=250)

    """
    Train DenoisingAutoEncoderSR => Did not work
    """

    # dsr = models.DenoisingAutoEncoderSR(scale)
    # dsr.create_model()
    # dsr.fit(nb_epochs=250)

    """
    Train Deep Denoise SR => Did not work
    """

    # ddsr = models.DeepDenoiseSR(scale)
    # ddsr.create_model()
    # ddsr.fit(nb_epochs=180)

    """
    Train Res Net SR => Did work
    """

    # rnsr = models.ResNetSR(scale)
    # rnsr.create_model(load_weights=True)
    # rnsr.fit(nb_epochs=50)

    """
    Train ESPCNN SR => Did not work
    """

    # espcnn = models.EfficientSubPixelConvolutionalSR(scale)
    # espcnn.create_model()
    # espcnn.fit(nb_epochs=50)

    """
    Train GAN Super Resolution => Did not work
    """

    # gsr = models.GANImageSuperResolutionModel(scale)
    # gsr.create_model(mode='train')
    # gsr.fit(nb_pretrain_samples=10000, nb_epochs=10)

    """
    Train Non Local ResNets => Did work
    """

    # non_local_rnsr = models.NonLocalResNetSR(scale)
    # non_local_rnsr.create_model()
    # non_local_rnsr.fit(nb_epochs=50)

    """
    Evaluate Image Stitching
    """

    # sr = model_stitching.ImageStitchingModel()
    # sr.stitch(cfg.dataset_folder + "/images/2/tk_0.png")

    """
    Evaluate Image Stitching
    """

    # sr = model_stitching.ExpantionStitching()
    # sr.stitch(cfg.dataset_folder + "/images/2/tk_0.png")

    """
    Evaluate Image Stitching
    """

    # sr = model_stitching.DenoisingAutoEncoderStitch()
    # sr.stitch(cfg.dataset_folder + "/images/2/tk_0.png")

    """
    Evaluate Super Resolution on Set5/14
    """

    # sr = models.ImageSuperResolutionModel(scale)
    # sr.evaluate(val_path)

    """
    Evaluate ESRCNN on Set5/14
    """

    # esr = models.ExpantionSuperResolution(scale)
    # esr.evaluate(val_path)

    """
    Evaluate DSRCNN on Set5/14 cannot be performed at the moment.
    This is because this model uses Deconvolution networks, whose output shape must be pre determined.
    This causes the model to fail to predict different images of different image sizes.
    """

    # dsr = models.DenoisingAutoEncoderSR(scale)
    # dsr.evaluate(val_path)

    """
    Evaluate DDSRCNN on Set5/14
    """

    # ddsr = models.DeepDenoiseSR(scale)
    # ddsr.evaluate(val_path)

    """
    Evaluate ResNetSR on Set5/14
    """

    # rnsr = models.ResNetSR(scale)
    # rnsr.create_model(None, None, 3, load_weights=True)
    # rnsr.evaluate(val_path)

    """
    Distilled ResNetSR
    """

    # distilled_rnsr = models.DistilledResNetSR(scale)
    # distilled_rnsr.create_model(None, None, 3, load_weights=True)
    # distilled_rnsr.evaluate(val_path)

    """
    Evaluate ESPCNN SR on Set 5/14
    """

    # espcnn = models.EfficientSubPixelConvolutionalSR(scale)
    # espcnn.evaluate(val_path)

    """
    Evaluate GAN Super Resolution on Set 5/14
    """

    # gsr = models.GANImageSuperResolutionModel(scale)
    # gsr.evaluate(val_path)

    """
    Evaluate Non Local ResNetSR on Set 5/14
    """

    # non_local_rnsr = models.NonLocalResNetSR(scale)
    # non_local_rnsr.evaluate(val_path)

    """
    Compare output images of sr, esr, dsr and ddsr models
    """

    # sr = models.ImageSuperResolutionModel(scale)
    # sr.upscale(path, save_intermediate=False, suffix="sr")

    # esr = models.ExpantionSuperResolution(scale)
    # esr.upscale(path, save_intermediate=False, suffix="esr")

    # dsr = models.DenoisingAutoEncoderSR(scale)
    # dsr.upscale(path, save_intermediate=False, suffix="dsr")

    # ddsr = models.DeepDenoiseSR(scale)
    # ddsr.upscale(path, save_intermediate=False, suffix="ddsr")

    # rnsr = models.ResNetSR(scale)
    # rnsr.create_model(None, None, 3, load_weights=True)
    # rnsr.upscale(path, save_intermediate=False, suffix="rnsr")

    # gansr = models.GANImageSuperResolutionModel(scale)
    # gansr.upscale(path, save_intermediate=False, suffix='gansr')
