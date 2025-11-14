#!/bin/bash

cd pixel-perfect-depth 

# Define some dir vars 
BASE_DATASET_DIR=../data/hypersim_processed

TRAIN_DIR=$BASE_DATASET_DIR/train
TRAIN_INPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/train_in
TRAIN_OUTPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/train_out

VAL_DIR=$BASE_DATASET_DIR/val
VAL_INPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/val_in
VAL_OUTPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/val_out

TEST_DIR=$BASE_DATASET_DIR/test
TEST_INPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/test_in
TEST_OUTPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/test_out

# Create respective input/ouput dirs
mkdir -p $TRAIN_INPUT_DIR
mkdir -p $TRAIN_OUTPUT_DIR

mkdir -p $VAL_INPUT_DIR
mkdir -p $VAL_OUTPUT_DIR

mkdir -p $TEST_INPUT_DIR
mkdir -p $TEST_OUTPUT_DIR

# Isolate the rgb files from the train, val and test dirs as input for pixel perfect depth
cp $TRAIN_DIR/ai*/rgb* $TRAIN_INPUT_DIR
cp $VAL_DIR/ai*/rgb* $VAL_INPUT_DIR
cp $TEST_DIR/ai*/rgb* $TEST_INPUT_DIR

echo "Running pixel perfect depth on training split..."
python3 run.py --img_path $TRAIN_INPUT_DIR --outdir $TRAIN_OUTPUT_DIR --pred_only
echo "Running pixel perfect depth on validation split..."
python3 run.py --img_path $VAL_INPUT_DIR --outdir $VAL_OUTPUT_DIR --pred_only
echo "Running pixel perfect depth on testing split..."
python3 run.py --img_path $TEST_INPUT_DIR --outdir $TEST_OUTPUT_DIR --pred_only

# Zip train, val and test datasets for download
BASE_ZIPPED_OUTPUT_DIR=../data/pixel_perfect_depth/hypersim
mkdir -p $BASE_ZIPPED_OUTPUT_DIR

echo "Zipping pixel_perfect_depth datasplits..."
zip -r $BASE_ZIPPED_OUTPUT_DIR/train.zip $TRAIN_OUTPUT_DIR 
zip -r $BASE_ZIPPED_OUTPUT_DIR/val.zip $VAL_OUTPUT_DIR 
zip -r $BASE_ZIPPED_OUTPUT_DIR/test.zip $TEST_OUTPUT_DIR


