#!/bin/bash

source env/bin/activate

mkdir -p $BASE_DATA_DIR/hypersim

echo "Downloading hypersim dataset..."
python3 script/depth/dataset_preprocess/hypersim/dataset_download_images.py \
  --downloads_dir $BASE_DATA_DIR/hypersim \
  --decompress_dir $BASE_DATA_DIR/hypersim \
  --delete_archive_after_decompress

echo "Unzipping dataset..."
for f in $BASE_DATA_DIR/hypersim/*zip; do
  unzip -o "$f" -d "${f%.zip}"
done

echo "Preprocessing hypersim dataset..."
python3 script/depth/dataset_preprocess/hypersim/preprocess_hypersim.py \
  --split_csv data_split/hypersim_depth/metadata_images_split_scene_v1.csv \
  --dataset_dir $BASE_DATA_DIR/hypersim \
  --output_dir $BASE_DATA_DIR/hypersim_processed
