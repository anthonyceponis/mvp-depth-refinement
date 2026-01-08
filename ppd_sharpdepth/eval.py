from argparse import ArgumentParser
from src.dataset import get_dataset, DatasetMode
from pydantic import BaseModel
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from pprint import pprint
import subprocess
import math
from PIL import Image

from ppd_sharpdepth.depth_estimators import ModelArchitecture 
from script.evaluation.metrics import abs_rel, rmse, rmse_rel, dbe_completeness, ppd_metric

def save_numpy_bitmap_as_png(bitmap_array: np.ndarray, file_path: str):
    
    if bitmap_array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional. Found {bitmap_array.ndim} dimensions.")

    if bitmap_array.dtype == bool:
        processed_array = bitmap_array.astype(np.uint8) * 255
        
    elif np.max(bitmap_array) <= 1 and np.min(bitmap_array) >= 0:
        processed_array = (bitmap_array.astype(np.uint8) * 255)
        
    else:
        processed_array = bitmap_array.astype(np.uint8)


    img = Image.fromarray(processed_array, mode='L')

    try:
        img.save(file_path, 'PNG')
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    parser = ArgumentParser("Script to evaluate outputs of models against ground truth labels.")

    parser.add_argument("--dataset_config_path", type=str, required=True, help="Path of the dataset config yaml file.")
    parser.add_argument("--model_architecture", type=str, required=True, help="Model.")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size.")
    parser.add_argument("--run_message", type=str, default=None, help="Message for this eval run. If not provided, will prompt interactively.")
    parser.add_argument("--debug", action="store_true", help="Print per-image metrics for debugging/comparison.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name (determines the output directory).")
    args = parser.parse_args()

    dataset_config_path = args.dataset_config_path
    model_architecture = ModelArchitecture(args.model_architecture)
    run_message = args.run_message if args.run_message else input("Give a message for this eval run: ")
    debug = args.debug
    
    BASE_DATA_DIR = Path(os.environ["BASE_DATA_DIR"])
    BASE_PREDS_DIR = Path(os.environ["BASE_PREDS_DIR"])
    results_filepath = Path("results.csv")

    cfg_data = OmegaConf.load(dataset_config_path)
    
    dataset = get_dataset(
        cfg_data, base_data_dir=BASE_DATA_DIR, mode=DatasetMode.EVAL
    )

    if args.subset_size is not None:
        import random
        random.seed(42)  # Fixed seed for reproducibility between infer and eval
        idxes = random.sample(range(len(dataset)), min(args.subset_size, len(dataset)))
        from torch.utils.data import Subset
        dataset = Subset(dataset, idxes)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    model_architecture = ModelArchitecture(model_architecture) 
 
    abs_rel_values = []
    rmse_values = []
    rmse_rel_values = []
    ppd_values = []

    # computing edge metric requires intrinsic data and high quality edges (from synthetic data), and hypersim is the only supported dataset which meets these requirements.
    with_edge_metric = cfg_data.name == "hypersim_depth"

    for data in tqdm(dataloader, desc="Evaluating"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        model_name = args.model_name if args.model_name else model_architecture.value

        pred_path = BASE_PREDS_DIR / cfg_data.dir / model_name / (rgb_name + ".npy")
        depth_pred = np.load(str(pred_path)).astype(np.float32)
        
        depth_pred = np.squeeze(depth_pred)
        
        img_abs_rel = abs_rel(depth_pred, depth_raw, valid_mask)
        img_rmse = rmse(depth_pred, depth_raw, valid_mask)
        img_rmse_rel = rmse_rel(depth_pred, depth_raw, valid_mask)
        abs_rel_values.append(img_abs_rel)
        rmse_values.append(img_rmse)
        rmse_rel_values.append(img_rmse_rel)
        
        ppd_score = None
        if with_edge_metric:
            intrinsics_ts = data["intrinsics"].squeeze()
            intrinsics = intrinsics_ts.numpy()
            ppd_score = ppd_metric(depth_pred, depth_raw, intrinsics)
            if ppd_score > 0:
                ppd_values.append(ppd_score)
        
        if debug:
            ppde_str = f"{ppd_score:.4f}" if ppd_score is not None and ppd_score > 0 else ""
            print(f"model={model_name}, image={rgb_name}, abs_rel={img_abs_rel:.4f}, rmse={img_rmse:.4f}, ppde={ppde_str}")
        
        # DEBUG NORMAL MAPS VIA VISUALISATION.
        #predicted_edge_path = BASE_PREDS_DIR / cfg_data.dir / model_architecture.value / (rgb_name + "_pred_edges.png")
        #ground_edge_path = BASE_PREDS_DIR / cfg_data.dir / model_architecture.value / (rgb_name + "_ground_edges.png")

        #save_numpy_bitmap_as_png(ground_edges, str(ground_edge_path))
        #save_numpy_bitmap_as_png(predicted_edges, str(predicted_edge_path))

    print("Total number of samples evaluated on: ", len(abs_rel_values))

    gitname = subprocess.check_output(["git", "config", "user.name"]).decode().strip()
    
    abs_rel_arr = np.array(abs_rel_values)
    rmse_arr = np.array(rmse_values)
    rmse_rel_arr = np.array(rmse_rel_values)
    
    abs_rel_mean = np.mean(abs_rel_arr)
    abs_rel_std_err = np.std(abs_rel_arr, ddof=1) / np.sqrt(len(abs_rel_arr))
    
    rmse_mean = np.mean(rmse_arr)
    rmse_std_err = np.std(rmse_arr, ddof=1) / np.sqrt(len(rmse_arr))
    
    rmse_rel_mean = np.mean(rmse_rel_arr)
    rmse_rel_std_err = np.std(rmse_rel_arr, ddof=1) / np.sqrt(len(rmse_rel_arr))
    
    if ppd_values:
        ppd_arr = np.array(ppd_values)
        ppd_mean = np.mean(ppd_arr)
        ppd_std_err = np.std(ppd_arr, ddof=1) / np.sqrt(len(ppd_arr))
    else:
        ppd_mean = None
        ppd_std_err = None
    
    z = 1.96 # 95% ci
    abs_rel_ci = (abs_rel_mean - z * abs_rel_std_err, abs_rel_mean + z * abs_rel_std_err)
    rmse_ci = (rmse_mean - z * rmse_std_err, rmse_mean + z * rmse_std_err)
    if ppd_values:
        ppd_ci = (ppd_mean - z * ppd_std_err, ppd_mean + z * ppd_std_err)
    else:
        ppd_ci = (None, None)
    
    df = pd.read_csv(results_filepath)

    for col in ["abs_rel_std_err", "rmse_std_err", "rmse_rel", "rmse_rel_std_err", "ppd_std_err"]:
        if col not in df.columns:
            df[col] = None
    
    metrics = {
        "id": len(df[df["gitname"] == gitname]),
        "gitname": gitname,
        "model_architecture": model_architecture.value,
        "preds_dataset": cfg_data.dir,
        "run_message": run_message,
        "abs_rel": abs_rel_mean,
        "abs_rel_std_err": abs_rel_std_err,
        "rmse": rmse_mean,
        "rmse_std_err": rmse_std_err,
        "rmse_rel": rmse_rel_mean,
        "rmse_rel_std_err": rmse_rel_std_err,
        "ppd": ppd_mean,
        "ppd_std_err": ppd_std_err,
    }
    df = pd.concat([df, pd.DataFrame([metrics])])
    df.to_csv(results_filepath, index=False)

    pprint(metrics)

   
