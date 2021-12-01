#!/usr/bin/env bash

DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$DIRECTORY/libs
#echo "==>> $LIB_DIR"

## Data Test 1
#IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/SmartSysLab/cam0/
#OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/
#SCALE_FACTOR=3

# Data Test 2
# IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Malibu/Calibration/calib_img
OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/cvpr_result/Malibu
INPUT_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Malibu/Input/{camID:05d}/{imgID:05d}.jpg
CALIB_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Malibu/Input/{camID:05d}/{imgID:05d}.jpg
NBC=5 # Number of camera
NBI=400 # Maximum number of image to use
NBIS=10 # Maximum number of image to use for image stitching
SCALE_FACTOR=3

# # Data Test 3
# IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Street/Calibration
# OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/
# CALIB_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Street/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=14 # Number of camera
# NBI=486 # Maximum number of image to use
# SCALE_FACTOR=3

# # Data Test 4
# IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Terrace/Calibration
# OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/
# CALIB_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Terrace/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=14 # Number of camera
# NBI=430 # Maximum number of image to use
# SCALE_FACTOR=3

# # Data Test 5
# # IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Airplanes/Calibration
# OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/cvpr_result/Airplanes
# CALIB_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Airplanes/Input/{camID:05d}/{imgID:05d}.jpg
# INPUT_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Airplanes/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=5 # Number of camera
# NBI=599 # Maximum number of image to use (162)
# NBIS=30 # Maximum number of image to use for image stitching
# SCALE_FACTOR=3

# # Data Test 6 --> Not matching features for this dataset.
# IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Windmills/Calibration
# OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/
# CALIB_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Windmills/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=5 # Number of camera
# NBI=100 # Maximum number of image to use
# SCALE_FACTOR=3

# # Data Test 7
# # IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Opera/Calibration
# OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/cvpr_result/Opera
# CALIB_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Opera/Input/{camID:05d}/{imgID:05d}.jpg
# INPUT_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/Opera/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=5 # Number of camera
# NBI=450 # Maximum number of image to use for calibration
# NBIS=10 # Maximum number of image to use for image stitching
# SCALE_FACTOR=3

# # Data Test 8
# # IMGDIR=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/example-data/Campus
# OUTDIR=/home/smrtsyslab/projects/deepstitch-dataset/out_result/cvpr_result/Campus
# INPUT_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/example-data/Campus/CMU{camID:d}/medium{imgID:02d}.jpg
# CALIB_PATTERN=/home/smrtsyslab/projects/deepstitch-dataset/raw_data/example-data/Campus/CMU{camID:d}/medium{imgID:02d}.jpg
# NBC=3 # Number of camera
# NBI=13 # Maximum number of image to use
# NBIS=13 # Maximum number of image to use for image stitching
# SCALE_FACTOR=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR/pano &&
# Model: is, eis, dis, ddis, rnis, distilled_rnis (not yet trained)
# python main_stitching.py --imgdir $IMGDIR --outdir $OUTDIR --model rnis --calib_pattern $CALIB_PATTERN -nbc $NBC -nbi $NBI --scale_factor=$SCALE_FACTOR

python main_stitching.py --outdir $OUTDIR --model is --calib_pattern $CALIB_PATTERN --input_pattern $INPUT_PATTERN \
    -nbc $NBC -nbi $NBI -nbis $NBIS --scale_factor=$SCALE_FACTOR
