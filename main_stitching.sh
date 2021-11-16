#!/usr/bin/env bash

DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$DIRECTORY/libs
#echo "==>> $LIB_DIR"

## Data Test 1
#IMGDIR=/media/sf_Data/data_stitching/SmartSysLab/cam0/
#OUTDIR=/media/sf_Data/data_stitching/deepstitch-dataset/out_result/
#SCALE_FACTOR=1

## Data Test 2
#IMGDIR=/media/sf_Data/data_stitching/Malibu/Calibration/calib_img
#OUTDIR=/media/sf_Data/data_stitching/deepstitch-dataset/out_result/
#SCALE_FACTOR=3

## Data Test 3
#IMGDIR=/media/sf_Data/data_stitching/Street/Calibration
#OUTDIR=/media/sf_Data/data_stitching/deepstitch-dataset/out_result/
#SCALE_FACTOR=3

# Data Test 4
IMGDIR=/media/sf_Data/data_stitching/Terrace/Calibration
OUTDIR=/media/sf_Data/data_stitching/deepstitch-dataset/out_result/
SCALE_FACTOR=3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR/pano &&
# Model: is, eis, dis, ddis, rnis, distilled_rnis (not yet trained)
python main_stitching.py --imgdir $IMGDIR --outdir $OUTDIR --model rnis --scale_factor=$SCALE_FACTOR

