#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

CODE_DIR="/home/liuding/Documents/NLRN"
cd $CODE_DIR

#OUTPUT=$ARNOLD_OUTPUT
#FLIST="/mnt/cephfs/lab/liuding/Semantic_Segmentation/lists/train_combined_v4.list"
#ROOT_FOLDER="/mnt/cephfs/lab/liuding/data"
DATA_PATH="/home/liuding/Documents/NLRN/data_all.json"
LEARNING_RATE=1e-3
BATCH_SIZE=16
SIGMA=25

SCRIPT="train_v5.py"
DATA_NAME="data_v4"
MODEL_NAME="nlrn"
EXP_NO=1
STATE_NUM=12
PATCH_SIZE=43
TRAIN_DIR="checkpoints/$MODEL_NAME-$DATA_NAME-sigma$SIGMA-r$STATE_NUM-p$PATCH_SIZE"
#MODEL_FILE_IN="checkpoints/hairSegNet-200000"
#MODEL_FILE_IN="mdl/sigma25r12_nonLocal_v13_p43_rgb2gray/model_drrn_nonLocal_v13-data-try-2-25000"
MODEL_FILE_IN="checkpoints/model_nonLocal_v13-data_v4-sigma25/model_nonLocal_v13-data_v4-1-125000"
MODEL_FILE_OUT="$TRAIN_DIR/$MODEL_NAME-$DATA_NAME-$EXP_NO"
CONTINUE_TRAINING=False

ARGS="--data_name=$DATA_NAME --sigma=$SIGMA --state_num=$STATE_NUM --model_name=$MODEL_NAME --smoothed_loss_batch_num=1000 --model_file_out=$MODEL_FILE_OUT --snapshot_batch_num=25000 --model_file_in=$MODEL_FILE_IN --batch_size=$BATCH_SIZE --data_path=$DATA_PATH --learning_rate=$LEARNING_RATE --continue_training=$CONTINUE_TRAINING --crop_height=$PATCH_SIZE --crop_width=$PATCH_SIZE"

mkdir -p $TRAIN_DIR
rm "$TRAIN_DIR/*.py"
cp $SCRIPT $TRAIN_DIR
cp "models/$MODEL_NAME.py" $TRAIN_DIR
cp "data_providers/$DATA_NAME.py" $TRAIN_DIR
echo "GPU_ID: $CUDA_VISIBLE_DEVICES"
echo "$SCRIPT $ARGS"
python -u $SCRIPT $ARGS
#python3 -u $SCRIPT $ARGS
