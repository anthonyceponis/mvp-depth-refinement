# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import logging
import os
from pathlib import Path
from omegaconf import OmegaConf

from ppd_sharpdepth.ppd.utils.depth2pcd import depth2pcd
import open3d as o3d

os.environ["XFORMERS_DISABLED"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image 
from torchvision.transforms.functional import pil_to_tensor, resize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ppd_sharpdepth.sharpdepth.util.image_util import colorize_depth_maps, chw2hwc

import debugpy

from .depth_estimators import get_depth_estimator_fn, ModelArchitecture
from .preprocessors import MarigoldPreProcessor

from src.dataset import get_dataset, DatasetMode

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs."
    ) 

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of model.")
    parser.add_argument("--dataset_config_path", type=str, required=True, help="Path of the dataset config yaml file.")
    parser.add_argument("--model_architecture", type=str, required=True, help="Model.") 
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (determines the output directory).")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--make_point_cloud", action="store_true", help="Make point cloud.")
    #parser.add_argument("--input_dir", type=str, required=True, help="Input image dataset directory")
    #parser.add_argument("--output_dir", type=str, required=True, help="Output depth dataset directory.") 

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached")

    checkpoint_path = args.checkpoint
    dataset_config_path = args.dataset_config_path
    model_architecture = ModelArchitecture(args.model_architecture)
    half_precision = args.half_precision
    #output_dir = str(Path(os.environ["BASE_PREDS_DIR"]) / args.output_dir / model_architecture.value)
    #input_dir = str(Path(os.environ["BASE_DATA_DIR"]) / args.input_dir)

    #os.makedirs(output_dir, exist_ok=True)

    cfg_data = OmegaConf.load(dataset_config_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(f"Running with half precision ({dtype}), might lead to suboptimal result.")
    else:
        dtype = torch.float32
        variant = None

    model_infer_fn = get_depth_estimator_fn(model_architecture, device, dtype, checkpoint_path)
    
    BASE_DATA_DIR = Path(os.environ["BASE_DATA_DIR"])
    BASE_PREDS_DIR = Path(os.environ["BASE_PREDS_DIR"])

    input_dir = BASE_DATA_DIR / cfg_data.dir

    run_name = args.run_name if args.run_name else model_architecture.value

    output_dir = BASE_PREDS_DIR / cfg_data.dir / run_name

    color_map = "inferno_r"

    dataset = get_dataset(
        cfg_data, base_data_dir=BASE_DATA_DIR, mode=DatasetMode.EVAL
    )

    if args.subset_size is not None:
        import random
        random.seed(42)  # Fixed seed for reproducibility between infer and eval
        idxes = random.sample(range(len(dataset)), args.subset_size)
        from torch.utils.data import Subset
        dataset = Subset(dataset, idxes)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    for data in tqdm(dataloader, desc="Inferring"):
        # GT data
        rgb_raw_11chw = data["rgb_int"]
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        with torch.no_grad():
            depth_np_11hw = model_infer_fn(rgb_raw_11chw, MarigoldPreProcessor)
        depth_np_11hw = depth_np_11hw.cpu().numpy()
        save_path = output_dir / rgb_name
        os.makedirs(save_path.parent, exist_ok=True)
        np.save(save_path, depth_np_11hw)
 
        depth_np_11hw = np.squeeze(depth_np_11hw)

        if args.make_point_cloud:
            assert "intrinsics" in data, "intrinsics must be in data"
            intrinsic = data["intrinsics"].numpy().copy().squeeze()
            H, W = depth_np_11hw.shape[-2:]

            intrinsic[0, 0] *= W 
            intrinsic[1, 1] *= H
            intrinsic[0, 2] *= W
            intrinsic[1, 2] *= H

            rgb_raw_hwc = rgb_raw_11chw.squeeze(0).permute(1,2,0).cpu().numpy()

            pred_pcd = depth2pcd(depth_np_11hw.squeeze().squeeze(), intrinsic, color=rgb_raw_hwc, ret_pcd=True)
            o3d.io.write_point_cloud(save_path.parent / f"{save_path.stem}_pred_point_cloud.ply", pred_pcd)

            gt_pcd = depth2pcd(depth_raw.squeeze().squeeze(), intrinsic, color=rgb_raw_hwc, input_mask=valid_mask, ret_pcd=True)
            o3d.io.write_point_cloud(save_path.parent / f"{save_path.stem}_gt_point_cloud.ply", gt_pcd)
        
        depth_np_11hw_valid = depth_np_11hw * valid_mask
        depth_raw_valid = depth_raw * valid_mask

        depth_maps = {
            "pred_raw": depth_np_11hw,
            "label_raw": depth_raw,
            "pred_valid": depth_np_11hw_valid,
            "label_valid": depth_raw_valid,
            "diff": np.abs(depth_np_11hw_valid - depth_raw_valid)
        }

        # Generate and save color maps.
        for variant, depth_map in depth_maps.items():
            depth_colored = colorize_depth_maps(depth_map, 0, depth_map.max(), cmap=color_map).squeeze()
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_label_colored_img = Image.fromarray(depth_colored_hwc)

            depth_label_colored_img.save(f"{save_path.parent}/{save_path.stem}_depth_{variant}.png")
    
    print(f"successfully saved outputs.")


