# MVP-DEPTH-ESTIMATION-WITH-REFINEMENT

### Setup and visualizing outputs

```bash
source ./setup.sh # installs dependencies for repo and submodules
source ./script/data_fetch/data-fetch-small.sh # use data-fetch.sh for the full datasets
source ./script/data_fetch/construct_lists.sh # constructs data split txt files for train/val/test for each dataset.
```

### Inference

All models are abstracted into a single function in `ppd_sharpdepth/depth_estimators.py`. Examples of running inference on a dataset can be found in `infer.sh`. Model outputs are dumped into the `preds` directory. Note that the model_architecture must match an enum string from the ModelArchitecture Enum in `ppd_sharpdetph/depth_estimators.py`

### Evaluation

See `eval.sh` for examples of how to run inference for a given dataset.

Run this to infer the external version of the models on big datasets.

```bash
source ./script/external_models/run-depth-anything.sh
source ./script/external_models/run-depth-v2.sh
source ./script/external_models/run-ppd.sh
source ./script/external_models/run-sharpdepth.sh
```


### Docker container setup

First, make sure you have a docker hub account and have docker cli installed.

Then, login into docker in the cli using `sudo docker login --u <username>`

Build a image using `sudo docker build -t <docker_username>/<image_name>:latest .` in the project root directory (don't forget the dot at the end!).

Verify the image is built and on your system using `sudo docker images`

Push image to docker hub using `sudo docker push <docker_username>/<image_name>:latest`


### Inference with PatchRefiner

This is a hard-coded temporary fix for running inference on PatchRefiner:
After setting up using `source ./setup.sh`, replace `env/lib/python3.13/site-packages/mmengine/registry/registry.py` with `ppd_sharpdepth/patchrefiner/hardcode_changes/registry.py`.

Sometimes, you may reach a RuntimeError stating there is an error in `loading state_dict for DPTDepthModel` and there are `Unexpected key(s) in state_dict` (all ending in `relative_position_index`). If this occurs, please ensure that the `timm` version is `0.9.2`, since newer versions will not match the `state_dict` keys correctly.
