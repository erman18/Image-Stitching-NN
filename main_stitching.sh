#!/usr/bin/env bash

DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$DIRECTORY/libs
#echo "==>> $LIB_DIR"
#IMGDIR=/media/sf_Data/data_stitching/deepstitch-dataset/images/1
IMGDIR=/media/sf_Data/data_stitching/SmartSysLab

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR/pano &&
python main_stitching.py --imgdir $IMGDIR --model rnis --scale_factor=3

