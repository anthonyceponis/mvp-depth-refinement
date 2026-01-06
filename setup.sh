#!/bin/bash

git submodule update --init --recursive

python3 -m venv env

source env/bin/activate

source ./set-env.sh

echo "" >> ~/.bashrc
cat set-env.sh >> ~/.bashrc

mkdir -p $BASE_DATA_DIR

echo "installing top-level dependencies..."

pip install -qr requirements+++.txt -r requirements++.txt -r requirements+.txt -r requirements.txt

echo "done installing top-level dependencies."

pushd submodules/UniDepth > /dev/null

echo "installing UniDepth in top-level environment."
pip install -v -e . --no-deps

if ! python -c "import torch; import KNN" &>/dev/null; then
  echo "Installing a custom op used by UniDepth. This will build some C++ code"
  cd unidepth/ops/knn
  python setup.py build install
fi

popd > /dev/null

echo "setup complete!"
