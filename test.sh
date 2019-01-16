#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

SIGMA=25
TRAIN_DIR="checkpoints/sigma$SIGMA"
MODEL_NAME="nlrn"
DATA_NAME="data"
PATCH_SIZE=45
STATE_NUM=12
MODEL_FILE="$TRAIN_DIR/$MODEL_NAME-$DATA_NAME-400000"

# Test on Set12 or BSD68
IMAGE_FOLDER="./data/Set12/"
#IMAGE_FOLDER="./data/BSD68/"
NOISY_IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/Set12/sigma$SIGMA/"
ARGS="--data_name=$DATA_NAME --image_folder=$IMAGE_FOLDER --noisy_image_folder=$NOISY_IMAGE_FOLDER --model_name=$MODEL_NAME --model_file=$MODEL_FILE --sigma=$SIGMA --patch_size=$PATCH_SIZE --state_num=$STATE_NUM"

echo $ARGS
python test.py $ARGS
