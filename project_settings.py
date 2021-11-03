import logging

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging_format = '%(asctime)s: %(levelname)s:%(message)s'
base_folder = "/media/sf_Data/data_stitching/deepstitch-dataset"

project_dir = "."  # "/home/erman/projects/Image-Stitching-NN"
libs_dir = project_dir + "/libs"
pano_libs_dir = libs_dir + "/pano"
pano_config_file = project_dir + "/settings/config.cfg"

