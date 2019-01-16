#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

DATA_NAME="data"
FLIST="/ws/ifp-06_1/dingliu2/data/SR/BSDS500/data/images/train_test_rgb2gray.list"
ROOT_FOLDER="/ws/ifp-06_1/dingliu2/data/SR/BSDS500/data/images/"
LEARNING_RATE=1e-3
BATCH_SIZE=16
SCRIPT="train.py"
SIGMA=25
STATE_NUM=12
PATCH_SIZE=43
MODEL_NAME="nlrn"
TRAIN_DIR="checkpoints/$MODEL_NAME-$DATA_NAME-sigma$SIGMA-r$STATE_NUM-p$PATCH_SIZE"
MODEL_FILE_IN="mdl/sigma25r12_nonLocal_v27_p43_rgb2gray/model_drrn_nonLocal_v27-data-try-2-400000"
MODEL_FILE_OUT="$TRAIN_DIR/$MODEL_NAME-$DATA_NAME"
CONTINUE_TRAINING=False

ARGS="--data_name=$DATA_NAME --state_num=$STATE_NUM --flist=$FLIST --root_folder=$ROOT_FOLDER --model_name=$MODEL_NAME --learning_rate=$LEARNING_RATE --sigma=$SIGMA --model_file_in=$MODEL_FILE_IN --model_file_out=$MODEL_FILE_OUT --smoothed_loss_batch_num=1000 --snapshot_batch_num=25000 --patch_size=$PATCH_SIZE --batch_size=$BATCH_SIZE --continue_training=$CONTINUE_TRAINING"

mkdir -p $TRAIN_DIR
rm "$TRAIN_DIR/*.py"
cp $SCRIPT $TRAIN_DIR
cp "models/$MODEL_NAME.py" $TRAIN_DIR
cp "data_providers/$DATA_NAME.py" $TRAIN_DIR
echo "GPU_ID: $CUDA_VISIBLE_DEVICES"
echo "$SCRIPT $ARGS"
python -u $SCRIPT $ARGS
