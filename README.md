# Depth refinement with diffusion model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anthonyceponis/mvp-depth-refinement/blob/main/demo.ipynb)

We adapted the diffusion-based depth estimation model [Pixel-Perfect Depth]() to perform depth refinement.

You can run our trained model using the Colab notebook linked above.

## Setup 

```bash
source ./setup.sh # installs dependencies for repo and submodules
source ./script/data_fetch/data-fetch-small.sh # use data-fetch.sh for the full datasets
source ./script/data_fetch/construct_lists.sh # constructs data split txt files for train/val/test for each dataset.
```

## Inference

All models are abstracted into a single function in `ppd_sharpdepth/depth_estimators.py`. Examples of running inference on a dataset can be found in `infer.sh`. Model outputs are dumped into the `preds` directory. Note that the model_architecture must match an enum string from the ModelArchitecture Enum in `ppd_sharpdetph/depth_estimators.py`

## Evaluation

See `eval.sh` for examples for how to run inference for a given dataset.


## Architecture

<img width="1168" height="1231" alt="image" src="https://github.com/user-attachments/assets/b058c442-84de-4878-bc22-db555d3aaa94" />
