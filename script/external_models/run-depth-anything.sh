#!/bin/bash

cd Depth-Anything
source env/bin/activate

BASE_HYPERSIM_DATASET_DIR=../data/hypersim_processed

for SPLIT in train val test; do
    SPLIT_DIR="$BASE_HYPERSIM_DATASET_DIR/$SPLIT"

    for dir in "$SPLIT_DIR"/*/; do
        LABELS_DIR="${dir%/}_labels"
        mkdir $LABELS_DIR 
        mv "$dir"depth*  $LABELS_DIR
        echo "Running depth anything on $SPLIT split..."
        python3 run.py --encoder vits --img-path "$dir" --outdir "$dir" --pred-only
        mv "$LABELS_DIR"/depth* $dir
        rm -rf $LABELS_DIR
    done
done

echo "Zipping depth_anything processed hypersim..."
zip -r "$BASE_DATA_DIR/depth_anything_hypersim.zip" "$BASE_HYPERSIM_DATASET_DIR"

