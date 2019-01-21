#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

EXPR_NAME="try"
TRAIN_DIR="checkpoints/sr"
MODEL_NAME="nlrn"
DATA_NAME="data_sr"
SCALE=3
STATE_NUM=12
MODEL_FILE="$TRAIN_DIR/$MODEL_NAME-$DATA_NAME-400000"
IMAGE_FOLDER="./data/Set5/"
OUTPUT_PATH="./result_sr/Set5-x$SCALE/"
mkdir -p $OUTPUT_PATH

ARGS="--data_name=$DATA_NAME --image_folder=$IMAGE_FOLDER --scale=$SCALE --output_path=$OUTPUT_PATH --model_name=$MODEL_NAME --model_file=$MODEL_FILE"

echo $ARGS
python test_sr.py $ARGS
