#!/bin/bash

source set-env.sh

echo "Setting up current repo..."
python3 -m venv env
source env/bin/activate 
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt

echo "Fetching depth anything repo..."
git clone https://github.com/LiheYoung/Depth-Anything.git
cd Depth-Anything
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

echo "Fetching depth anything checkpoints..."
mkdir checkpoints
wget -O checkpoints/depth_anything_metric_depth_indoor.pth "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt?download=true"

# cd ..

# echo "Fetching pixel pixel perfect repo..."
# git clone https://github.com/gangweix/pixel-perfect-depth.git 
# cd pixel-perfect-depth
# python3 -m venv env
# source env/bin/activate
# pip install -r requirements.txt

# echo "Fetching pixel perfect checkpoints..."
# mkdir checkpoints
# wget -O checkpoints/ppd.pth "https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth"
# wget -O checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

