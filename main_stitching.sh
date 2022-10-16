#!/usr/bin/env bash

DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$DIRECTORY/libs
DFS="MCMI"
#echo "==>> $LIB_DIR"

# # Data Test 1
# IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/SmartSysLab/cam0/
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/SmartSysLab/cam0/
# SCALE_FACTOR=3

# # Data Test 2
# # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Malibu/Calibration/calib_img
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Malibu/3
# INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Malibu/Input/{camID:05d}/{imgID:05d}.jpg
# CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Malibu/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=5 # Number of camera
# NBI=400 # Maximum number of image to use for calibration
# NBIS=10 # Maximum number of image to use for image stitching
# SCALE_FACTOR=4

# # Data Test 3
# # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Street/Calibration
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Street/latest/3
# INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Street/Input/{camID:05d}/{imgID:05d}.jpg
# CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Street/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=14 # Number of camera
# NBI=486 # Maximum number of image to use for calibration
# NBIS=10 # Maximum number of image to use for image stitching
# SCALE_FACTOR=4

# # Data Test 4
# # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Terrace/Calibration
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Terrace3/3
# CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Terrace/Input/{camID:05d}/{imgID:05d}.jpg
# INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Terrace/Input/{camID:05d}/{imgID:05d}.jpg
# DFS="MCMI"
# NBC=14 # Number of camera
# NBIS=10 # Maximum number of image to use for image stitching
# NBI=430 # Maximum number of image to use for calibration
# SCALE_FACTOR=3

# # Data Test 5
# # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Airplanes/Calibration
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Airplanes2/3
# CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Airplanes/Input/{camID:05d}/{imgID:05d}.jpg
# INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Airplanes/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=5 # Number of camera
# NBI=599 # Maximum number of image to use for calibration (162)
# NBIS=10 # Maximum number of image to use for image stitching
# SCALE_FACTOR=3

# # Data Test 6 --> Not matching features for this dataset. (GAUSS_WINDOW_FACTOR 3, RANSAC_INLIER_THRES 2.5)
# # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Windmills/Calibration
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Windmills2/3
# INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Windmills/Input/{camID:05d}/{imgID:05d}.jpg
# CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Windmills/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=5 # Number of camera
# NBI=300 # Maximun number of images to use for calibration
# NBIS=10 # Maximum number of image to use for image stitching
# SCALE_FACTOR=4

# # Data Test 7
# # IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/Opera/Calibration
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/Opera/3
# CALIB_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Opera/Input/{camID:05d}/{imgID:05d}.jpg
# INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/Opera/Input/{camID:05d}/{imgID:05d}.jpg
# NBC=5 # Number of camera
# NBI=450 # Maximum number of image to use for calibration
# NBIS=10 # Maximum number of image to use for image stitching
# SCALE_FACTOR=3

# # Data Test 8
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/CMU0/2
# INPUT_PATTERN=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/Campus/CMU0/medium{imgID:02d}.jpg
# NBIS=37 # Maximum number of image to use for image stitching
# DFS="SCMI"
# SCALE_FACTOR=2

# Data Test 9
IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/Campus/CMU1
OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/CMU1/3
DFS="IDIR"
SCALE_FACTOR=2

# # Data Test 10
# IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/UFCampus/UFWEST4
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/UFWEST4/3
# DFS="IDIR"
# SCALE_FACTOR=3

# # Data Test 11
# IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/zijing
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/zijing/3
# DFS="IDIR"
# SCALE_FACTOR=3

# # Data Test 12
# IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/flower
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/flower/3
# DFS="IDIR"
# SCALE_FACTOR=1

# # Data Test 13
# IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/example-data/NSH
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/NSH/3
# DFS="IDIR"
# SCALE_FACTOR=3

# # Data Test 14
# IMGDIR=/mnt/data/enghonda/deepstitch-dataset/raw_data/UFCampus/LAR230LAB
# OUTDIR=/mnt/data/enghonda/deepstitch-dataset/cvpr_result/LAR230LAB/3
# DFS="IDIR"
# SCALE_FACTOR=3

METRIC=lpips
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR/pano &&
# Model: is, eis, dis, ddis, rnis, distilled_rnis (not yet trained)
# New model: unrnis, unddis
python main_stitching.py --imgdir $IMGDIR --outdir $OUTDIR --model unrnis --calib_pattern $CALIB_PATTERN --input_pattern $INPUT_PATTERN \
    -nbc $NBC -nbi $NBI -nbis $NBIS --scale_factor=$SCALE_FACTOR --dfs=$DFS --files=$FILES --metric=$METRIC
