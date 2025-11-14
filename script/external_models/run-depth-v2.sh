# Fetch repo
cd ..
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Fetch depth anything v2 small checkpoint
mkdir checkpoints
wget -O checkpoints/depth_anything_v2_vits.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"

# Define some dir vars 
$BASE_DATASET_DIR = ../mvp-depth-refinement/data/hypersim_processed

$TRAIN_DIR = $BASE_DATASET_DIR/train
$TRAIN_INPUT_DIR = $BASE_DATASET_DIR/train_in
$TRAIN_OUTPUT_DIR = $BASE_DATASET_DIR/train_out

$VAL_DIR = $BASE_DATASET_DIR/val
$VAL_INPUT_DIR = $BASE_DATASET_DIR/val_in
$VAL_OUTPUT_DIR = $BASE_DATASET_DIR/val_out

$TEST_DIR = $BASE_DATASET_DIR/test
$TEST_INPUT_DIR = $BASE_DATASET_DIR/test_in
$TEST_OUTPUT_DIR = $BASE_DATASET_DIR/test_out

# Create respective input/ouput dirs
mkdir $TRAIN_DIR_INPUT
mkdir $TRAIN_DIR_OUTPUT

mkdir $VAL_DIR_INPUT
mkdir $VAL_DIR_OUTPUT

mkdir $TEST_DIR_INPUT
mkdir $TEST_DIR_OUTPUT

# Isolate the rgb files from the train, val and test dirs as input for depth anything
cp $TRAIN_DIR/rgb* $TRAIN_DIR_INPUT
cp $VAL_DIR/rgb* $VAL_DIR_INPUT
cp $TEST_DIR/rgb* $TEST_DIR_INPUT

# Run depth anything on train, val and test datasets
python run.py --encoder vits --img-path $TRAIN_INPUT_DIR --outdir $TRAIN_OUTPUT_DIR
python run.py --encoder vits --img-path $VAL_INPUT_DIR --outdir $VAL_OUTPUT_DIR
python run.py --encoder vits --img-path $TEST_INPUT_DIR --outdir $TEST_OUTPUT_DIR

# Zip train, val and test datasets for download
