#!/bin/bash

cd Depth-Anything

# Define some dir vars 
BASE_DATASET_DIR=../data/hypersim_processed

TRAIN_DIR=$BASE_DATASET_DIR/train
TRAIN_INPUT_DIR=$BASE_DATASET_DIR/depth_anything/train_in
TRAIN_OUTPUT_DIR=$BASE_DATASET_DIR/depth_anything/train_out

VAL_DIR=$BASE_DATASET_DIR/val
VAL_INPUT_DIR=$BASE_DATASET_DIR/depth_anything/val_in
VAL_OUTPUT_DIR=$BASE_DATASET_DIR/depth_anything/val_out

TEST_DIR=$BASE_DATASET_DIR/test
TEST_INPUT_DIR=$BASE_DATASET_DIR/depth_anything/test_in
TEST_OUTPUT_DIR=$BASE_DATASET_DIR/depth_anything/test_out

# Create respective input/ouput dirs
mkdir -p $TRAIN_INPUT_DIR
mkdir -p $TRAIN_OUTPUT_DIR

mkdir -p $VAL_INPUT_DIR
mkdir -p $VAL_OUTPUT_DIR

mkdir -p $TEST_INPUT_DIR
mkdir -p $TEST_OUTPUT_DIR

# Isolate the rgb files from the train, val and test dirs as input for depth anything
cp $TRAIN_DIR/ai*/rgb* $TRAIN_INPUT_DIR
cp $VAL_DIR/ai*/rgb* $VAL_INPUT_DIR
cp $TEST_DIR/ai*/rgb* $TEST_INPUT_DIR

echo "Running depth anything on training split..."
python3 run.py --encoder vits --img-path $TRAIN_INPUT_DIR --outdir $TRAIN_OUTPUT_DIR --pred-only
echo "Running depth anything on validation split..."
python3 run.py --encoder vits --img-path $VAL_INPUT_DIR --outdir $VAL_OUTPUT_DIR --pred-only
echo "Running depth anything on testing split..."
python3 run.py --encoder vits --img-path $TEST_INPUT_DIR --outdir $TEST_OUTPUT_DIR --pred-only

# Zip train, val and test datasets for download
BASE_ZIPPED_OUTPUT_DIR=../data/depth_anything/hypersim
mkdir -p $BASE_ZIPPED_OUTPUT_DIR

echo "Zipping depth_anything datasplits..."
zip -r $BASE_ZIPPED_OUTPUT_DIR/train.zip $TRAIN_OUTPUT_DIR 
zip -r $BASE_ZIPPED_OUTPUT_DIR/val.zip $VAL_OUTPUT_DIR 
zip -r $BASE_ZIPPED_OUTPUT_DIR/test.zip $TEST_OUTPUT_DIR


