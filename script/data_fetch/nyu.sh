DATA_DIR="data"

echo "Fetching NYU V2 dataset..."
mkdir $DATA_DIR/nyu_v2_raw
wget -O $DATA_DIR/nyu_v2_raw/nyu_depth_v2_labeled.mat "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

echo "Preprocessing NYU V2 dataset..."
python3 script/depth/dataset_preprocess/nyu/nyu_preprocess.py
