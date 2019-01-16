#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

#EXPR_NAME="try"
#EXPR_NAME="train300"
TRAIN_DIR="checkpoints/sigma15"
MODEL_NAME="nlrn"
DATA_NAME="data"
SIGMA=15
PATCH_SIZE=45
STATE_NUM=12
MODEL_FILE="$TRAIN_DIR/$MODEL_NAME-$DATA_NAME-1975000"
#MODEL_FILE="$TRAIN_DIR/model_drrn_nonLocal_v7-data-try-1-225000"

# Test on Set12
IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/Set12/"
NOISY_IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/Set12/sigma$SIGMA/"
ARGS="--data_name=$DATA_NAME --image_folder=$IMAGE_FOLDER --noisy_image_folder=$NOISY_IMAGE_FOLDER --model_name=$MODEL_NAME --model_file=$MODEL_FILE --sigma=$SIGMA --patch_size=$PATCH_SIZE --state_num=$STATE_NUM"

echo $ARGS
python test.py $ARGS

exit

# Test on BSD68
IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/BSD68/"
NOISY_IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/BSD68/sigma$SIGMA/"
ARGS="--data_name=$DATA_NAME --image_folder=$IMAGE_FOLDER --noisy_image_folder=$NOISY_IMAGE_FOLDER --model_name=$MODEL_NAME --model_file=$MODEL_FILE"

echo $ARGS
#python validate.py $ARGS
python validate_patch.py $ARGS

# Test on S14
IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/S14/"
NOISY_IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/S14/sigma$SIGMA-seed54/"
ARGS="--data_name=$DATA_NAME --image_folder=$IMAGE_FOLDER --noisy_image_folder=$NOISY_IMAGE_FOLDER --model_name=$MODEL_NAME --model_file=$MODEL_FILE"

#echo $ARGS
#python validate.py $ARGS
#python validate_patch.py $ARGS

# Test on BSD200
#IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/BSD200_test_gray/"
#NOISY_IMAGE_FOLDER="/ws/ifp-06_1/dingliu2/data/Denoising/BSD200_test_gray/sigma$SIGMA/"
IMAGE_FOLDER="/media/dingliu2/3612CF7C12CF3F9B/MemNet/data/GaussianDenoising/BSD200_halfsize/"
NOISY_IMAGE_FOLDER="/media/dingliu2/3612CF7C12CF3F9B/MemNet/data/GaussianDenoising/BSD200_halfsize/sigma$SIGMA/"
ARGS="--data_name=$DATA_NAME --image_folder=$IMAGE_FOLDER --noisy_image_folder=$NOISY_IMAGE_FOLDER --model_name=$MODEL_NAME --model_file=$MODEL_FILE"

#echo $ARGS
#python validate.py $ARGS
#python validate_patch_v3.py $ARGS

# Test on Urban100
IMAGE_FOLDER="/media/dingliu2/3612CF7C12CF3F9B/data/Denoising/urban100_SRF4seed/"
NOISY_IMAGE_FOLDER=$IMAGE_FOLDER
ARGS="--data_name=$DATA_NAME --image_folder=$IMAGE_FOLDER --noisy_image_folder=$NOISY_IMAGE_FOLDER --model_name=$MODEL_NAME --model_file=$MODEL_FILE --sigma=$SIGMA"

#echo $ARGS
#python validate.py $ARGS
#python validate_patch_v2.py $ARGS
