import logging

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging_format = '%(asctime)s: %(levelname)s:%(message)s'
dataset_folder = "/home/smrtsyslab/projects/deepstitch-dataset"
image_folder = f"{dataset_folder}/training_data"

project_dir = "."  # "/home/erman/projects/Image-Stitching-NN"
project_setting_dir = project_dir + "/settings"
libs_dir = project_dir + "/libs"
log_dir = project_dir + "/logs"
pano_libs_dir = libs_dir + "/pano"
pano_config_file = project_setting_dir + "/config.cfg"

# Data preparation settings
patch_size = 256
patch_step = 64
sandfall_layer = -5

config_img_input = project_setting_dir + "/config_input_file.json"
config_img_output = project_setting_dir + "/config_output_file.json"
