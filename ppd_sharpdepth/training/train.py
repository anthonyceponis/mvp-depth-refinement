# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import itertools
import logging
import math
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import debugpy
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from ppd_sharpdepth.depth_estimators import ModelArchitecture, get_depth_estimator_fn
from ppd_sharpdepth.preprocessors import MarigoldPreProcessor, PixelPerfectDepthPreProcessor
from ppd_sharpdepth.sharpdepth.util.alignment import align_depth_least_square
import enum

import subprocess
import json
from rich.pretty import pprint
import sys
import shlex
import tempfile

from ppd_sharpdepth.sharpdepth.util.image_util import chw2hwc, colorize_depth_maps
from script.evaluation.metrics import rmse

os.environ["XFORMERS_DISABLED"] = "1"

from diffusers.models.attention_processor import AttnProcessor2_0
import diffusers
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from ema_pytorch import EMA
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from ppd_sharpdepth.base_depth_estimators import get_base_depth_estimator_fn
from ppd_sharpdepth.ppd.utils.transform import cv2_interpolate, resize_keep_aspect
from unidepth.models import UniDepthV1
import random
from ppd_sharpdepth.sharpdepth_kinds import SharpDepthKind

from ppd_sharpdepth.sharpdepth.data.datasets_and_samplers import BaseDepthDataset, DatasetMode, get_dataset
from ppd_sharpdepth.sharpdepth.data.datasets_and_samplers.mixed_sampler import MixedBatchSampler
from ppd_sharpdepth.sharpdepth.pipeline.pipeline import SharpDepthPipeline
from ppd_sharpdepth.sharpdepth.util.config_util import find_value_in_omegaconf, recursive_load_config
from ppd_sharpdepth.sharpdepth.util.logging_util import config_logging
from ppd_sharpdepth.sharpdepth.util.normalizer import ScaleShiftNormalizer
from ppd_sharpdepth.ppd.models.ppd import PixelPerfectDepth

import torchvision.transforms.functional as TV_F

import wandb
import cv2

logger = get_logger(__name__)

import torch
import torch.nn.functional as F

def depth_to_normals(depth: torch.Tensor, K: torch.Tensor):
    """
    depth: (B,1,H,W)
    K:     (B,3,3) or (3,3)
    returns (B,3,H,W) normals
    """
    B, _, H, W = depth.shape

    # handle K batch or single
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)

    fx = K[:, 0, 0].view(B,1,1,1)
    fy = K[:, 1, 1].view(B,1,1,1)
    cx = K[:, 0, 2].view(B,1,1,1)
    cy = K[:, 1, 2].view(B,1,1,1)

    # pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing="ij"
    )
    x = x.float()[None, None]
    y = y.float()[None, None]

    # backproject
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    # gradients
    def dx(t): return F.pad(t[..., 1:] - t[..., :-1], (1,0,0,0))
    def dy(t): return F.pad(t[..., 1:, :] - t[..., :-1, :], (0,0,1,0))

    Xx, Xy = dx(X), dy(X)
    Yx, Yy = dx(Y), dy(Y)
    Zx, Zy = dx(Z), dy(Z)

    tx = torch.cat([Xx, Yx, Zx], dim=1)
    ty = torch.cat([Xy, Yy, Zy], dim=1)

    normals = torch.cross(tx, ty, dim=1)
    normals = F.normalize(normals, dim=1)

    return normals


@torch.no_grad()
def encode_image(vae, rgb):
    rgb_latent = vae.encode(rgb).latent_dist.mean
    rgb_latent = rgb_latent * vae.config.scaling_factor
    return rgb_latent


@torch.no_grad()
def decode_image(vae, latent):
    # scale latent
    latent = latent / vae.config.scaling_factor
    # decode
    z = vae.post_quant_conv(latent)
    rgb = vae.decoder(z)
    return rgb


@torch.no_grad()
def encode_depth(vae, depth):
    depth_latent = vae.encode(depth.repeat(1, 3, 1, 1)).latent_dist.mean
    depth_latent = depth_latent * vae.config.scaling_factor
    return depth_latent


def l1_loss(pred_depth, gt_depth, mask):
    l1_loss = torch.abs(pred_depth - gt_depth) * mask
    l1_loss = l1_loss.sum() / (mask.sum() + 1e-8)
    return l1_loss


def abs_relative_difference_full(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    return abs_relative_diff


def encode_empty_text(tokenizer, text_encoder):
    """
    Encode text embedding for empty prompt
    """
    prompt = ""
    text_inputs = tokenizer(
        prompt,
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    empty_text_embed = text_encoder(text_input_ids)[0]

    return empty_text_embed


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def colorize(value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img

def auto_canny_depth_otsu(depth_map, apertureSize=3, L2gradient=False, dilate_kernel=0, low_frac=0.5):
    """
    Automatic Canny edge detection using Otsu thresholding on gradient magnitudes.

    Parameters:
        depth_map : 2D numpy array (depth values, any range)
        apertureSize : Sobel kernel size (3,5,7)
        L2gradient : whether to use L2 norm for gradient magnitude
        dilate_kernel : optional, size of square kernel to dilate edges after detection
        low_frac : fraction of high threshold to use as low threshold (default 0.5)

    Returns:
        edges : binary edge map (uint8, 0 or 255)
        threshold1, threshold2 : thresholds used
    """
    # 1. Normalize depth map to 0-255 uint8
    depth_norm = np.clip(depth_map, np.min(depth_map), np.max(depth_map))
    depth_uint8 = ((depth_norm - np.min(depth_norm)) / (np.max(depth_norm) - np.min(depth_norm)) * 255).astype(np.uint8)

    # 2. Compute gradient magnitude using Sobel
    grad_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=apertureSize)
    grad_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=apertureSize)
    
    if L2gradient:
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        grad_mag = np.abs(grad_x) + np.abs(grad_y)

    # 3. Otsu threshold on gradient magnitude
    grad_uint8 = np.clip((grad_mag / grad_mag.max() * 255), 0, 255).astype(np.uint8)
    threshold2, _ = cv2.threshold(grad_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to native Python floats
    threshold2 = float(threshold2)
    threshold1 = float(low_frac * threshold2)

    # 4. Run Canny
    edges = cv2.Canny(depth_uint8, threshold1, threshold2,
                     apertureSize=apertureSize, L2gradient=L2gradient)

    # 5. Optional dilation
    if dilate_kernel and dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel)

    return edges


@torch.no_grad()
def get_dilated_edge_mask(depth_11hw: torch.Tensor, distance_threshold_px: float = 30.0):
    assert depth_11hw.ndim == 4 and depth_11hw.shape[0] == 1 and depth_11hw.shape[1] == 1
    depth_hw_np = depth_11hw.squeeze(0,1).float().cpu().numpy()
    edges_np = auto_canny_depth_otsu(depth_hw_np)

    dist_from_edges = cv2.distanceTransform(255 - edges_np, cv2.DIST_L2, 5)

    is_near_edge_mask_np = dist_from_edges < distance_threshold_px
    is_near_edge_mask_11hw = torch.from_numpy(is_near_edge_mask_np).to(depth_11hw.device).unsqueeze(0).unsqueeze(0)
    assert is_near_edge_mask_11hw.shape == depth_11hw.shape
    return is_near_edge_mask_11hw


if "__main__" == __name__:
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your little sharpener!")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to save checkpoints"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--base_data_dir", type=str, default=None, help="directory of training data"
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="directory of pretrained checkpoint",
    )
    parser.add_argument(
        "--student_ckpt_dir",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="directory of pretrained checkpoint",
    )
    parser.add_argument("--student_ckpt_dir_revision", type=str, default=None, help="Revision of the student checkpoint HF Hub model")
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--depth_weight",
        type=float,
        default=0.2,
        help="Depth loss weight.",
    )
    parser.add_argument(
        "--normal_loss_weight",
        type=float,
        default=0.2,
        help="Depth loss weight.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="marigold_train_t2i_adapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--base_model", type=str, default="unidepth", help="Base model to use for depth estimation. Options: unidepth, depth_anything_small, depth_anything_large, pixel_perfect_depth")
    parser.add_argument("--denoiser", type=str, default="lotus", help="Which model constitutes SharpDepth. Options: lotus, pixel_perfect_depth, pixel_perfect_depth_controlnet")
    parser.add_argument("--use_conditioning_probability", type=float, default=0.8, help="Probability of using conditioning in the student denoiser")
    parser.add_argument("--wandb_name", type=str, default="", help="Name of the wandb run")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--debug_port", type=int, default=5678, help="Port for the debugger")
    parser.add_argument("--compute_initial_depth_loss_probability", type=float, default=1.0, help="Probability of computing the initial depth loss")
    parser.add_argument("--dit_patch_encoder_lr_multiplier", type=float, default=0.01, help="Multiplier for the learning rate of the DiT patch encoder")
    parser.add_argument("--sds_loss_weight", type=float, default=1.0, help="Weight for the SDS loss")
    parser.add_argument("--blur_unidepth_output_ratio", type=int, default=1, help="Ratio of downscaling the unidepth output, for guidance, l1 error, and depth loss computation")
    parser.add_argument("--noise_aware_latent_noise_scale", type=float, default=1.0, help="Scale for the noise aware latent noise. 0 means no noise, 1 means default behavior.")
    parser.add_argument("--use_conditioning_for_initial_ppd", action="store_true", help="Whether to use conditioning for the initial PPD")
    parser.add_argument("--gaussian_blur", action="store_true", help="Whether to use Gaussian blur for the difference map")
    parser.add_argument("--blur_depth_loss", action="store_true", help="Whether to blur the depth loss")
    parser.add_argument("--forward_diffuse_from",type=str, default="initial_pred_depth", help="Where to start the *forward* diffusion from during training. Options: initial_pred_depth, base_pred_depth")
    parser.add_argument("--log_depth_maps", action="store_true", help="Log training depth maps (SDS loss, depth loss, etc.) to disk. Slows down training")
    parser.add_argument("--initialize_ppd_from_timestep", type=int, default=None, help="Timestep to initialize the PPD from")
    parser.add_argument("--max_sds_timestep", type=int, default=None, help="Maximum timestep for the SDS loss")
    parser.add_argument("--align_depth_least_square", action="store_true", help="Whether to align the depth using least square")
    parser.add_argument("--flip_sign_for_controlnet", action="store_true", help="Whether to flip the sign for the controlnet")
    parser.add_argument("--depth_loss_away_from_edges_threshold_px", type=int, default=30, help="Threshold in pixels for the depth loss away from edges")
    parser.add_argument("--use_synthetic_conditioning_probability", type=float, default=1.0, help="Probability of using synthetic conditioning (for PPD_controlnet)")
    parser.add_argument("--forward_diffuse_from_initial_pred_depth_probability", type=float, default=1.0, help="Probability of using initial pred depth for forward diffusion. Only used if forward_diffuse_from is initial_pred_depth.")
    parser.add_argument("--edge_loss_blur_radius_px", type=int, default=8, help="Radius in pixels for the edge loss blur")
    parser.add_argument("--use_edge_loss_as_sds_loss", action="store_true", help="Whether to use the edge loss as the SDS loss")
    parser.add_argument("--use_sharpdepth_style_losses", action="store_true", help="Whether to use the sharpdepth style losses")
    parser.add_argument("--use_normal_loss", action="store_true", help="Whether to use surface normal consistency loss.")
    parser.add_argument("--use_public_pretrained_sharpdepth_weights", action="store_true", help="Whether to use surface normal consistency loss.")

    args = parser.parse_args()

    class ForwardDiffuseFrom(enum.Enum):
        INITIAL_PRED_DEPTH = "initial_pred_depth"
        BASE_PRED_DEPTH = "base_pred_depth"

    forward_diffuse_from = ForwardDiffuseFrom(args.forward_diffuse_from)

    sharpdepth_kind = SharpDepthKind(args.denoiser)

    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir if args.base_data_dir is not None else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir if args.base_ckpt_dir is not None else os.environ["BASE_CKPT_DIR"]
    )
    student_ckpt_dir = args.student_ckpt_dir
    student_ckpt_dir_revision = args.student_ckpt_dir_revision

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # ---------------------------------------------------------------------

    # -------------------- Initialization --------------------
    cfg = recursive_load_config(args.config)
    # Full job name
    pure_job_name = os.path.basename(args.config).split(".")[0]
    # Add time prefix
    if args.add_datetime_prefix:
        job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
    else:
        job_name = pure_job_name
    # ---------------------------------------------------------------------

    # -------------------- Initialize Logger and Accelerator --------------------
    logging_dir = Path(output_dir, job_name)
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # ---------------------------------------------------------------------

    if args.debug and accelerator.is_local_main_process:
        debugpy.listen(args.debug_port)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached")


    # -------------------- set seed --------------------
    if args.seed is not None:
        set_seed(args.seed)
    # ---------------------------------------------------------------------

    # -------------------- create logging folder --------------------
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
    # ---------------------------------------------------------------------

    # -------------------- unwrap function --------------------
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ---------------------------------------------------------------------

    # -------------------- create custom loading hooks --------------------
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                data = {"ema": ema_model.state_dict()}
                torch.save(data, os.path.join(output_dir, "denoiser_ema.pt"))
                del data

            for model in models:
                sub_dir = (
                    (denoiser_subfolder if model == unwrap_model(student_denoiser) else frozen_denoiser_subfolder)
                    if isinstance(model, type(unwrap_model(student_denoiser)))
                    else "text_encoder"
                )
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            data = torch.load(os.path.join(input_dir, "denoiser_ema.pt"), map_location="cpu")
            ema_model.load_state_dict(data["ema"], strict=False)
            del data

        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = denoiser_cls.from_pretrained(input_dir, subfolder=denoiser_subfolder)
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    # ---------------------------------------------------------------------

    # -------------------- Logging settings --------------------
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        config_logging(cfg.logging, out_dir=logging_dir)
        logger.info(f"config: {cfg}")
    # ---------------------------------------------------------------------

    # -------------------- Set training device --------------------
    device = accelerator.device
    logger.info(f"device = {device}")
    # ---------------------------------------------------------------------

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = args.train_batch_size * args.gradient_accumulation_steps
    logger.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {args.gradient_accumulation_steps}"
    )
    # ---------------------------------------------------------------------

    # -------------------- Create dataset --------------------
    if args.seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(args.seed)

    cfg_data = cfg.dataset

    # Training dataset
    train_dataset: BaseDepthDataset = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        # mode=DatasetMode.TRAIN,
        mode=DatasetMode.RGB_ONLY,
        augmentation_args=cfg.augmentation,
    )

    class IndexedDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            return self.dataset[index], index

    if "mixed" == cfg_data.train.name:
        for data in train_dataset:
            if len(data) == 0:
                breakpoint()
        dataset_ls = train_dataset
        if len(cfg_data.train.prob_ls) > 0:
            assert len(cfg_data.train.prob_ls) == len(
                dataset_ls
            ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        concat_dataset_with_idx = IndexedDataset(concat_dataset)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=args.train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_dataloader = DataLoader(
            concat_dataset_with_idx,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_train_workers,
        )
    elif "concat" == cfg_data.train.name:
        concat = ConcatDataset(train_dataset)
        concat_with_idx = IndexedDataset(concat)

        train_dataloader = DataLoader(
            concat_with_idx,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_train_workers,
            generator=loader_generator
        )

    # Validation dataset
    val_loaders: List[DataLoader] = []
    for _val_dic in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_val_workers,
        )
        val_loaders.append(_val_loader)
    logger.info("Finish loading dataset")
    # ---------------------------------------------------------------------

    # -------------------- Model --------------------
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, subfolder="tokenizer")
    text_encoder_cls = import_model_class_from_model_name_or_path(base_ckpt_dir, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(
        base_ckpt_dir, subfolder="text_encoder", revision=None
    )
    noise_scheduler = DDPMScheduler.from_pretrained(base_ckpt_dir, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(base_ckpt_dir, subfolder="vae", revision=None)

    frozen_denoiser_subfolder = {
        SharpDepthKind.LOTUS: "unet",
        SharpDepthKind.PIXEL_PERFECT_DEPTH: "ppd",
        SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET: "ppd"
    }[sharpdepth_kind]

    denoiser_subfolder = {
        SharpDepthKind.LOTUS: "unet_student",
        SharpDepthKind.PIXEL_PERFECT_DEPTH: "ppd_student",
        SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET: "ppd_student_controlnet"
    }[sharpdepth_kind]
    denoiser_cls = {
        SharpDepthKind.LOTUS: UNet2DConditionModel,
        SharpDepthKind.PIXEL_PERFECT_DEPTH: PixelPerfectDepth,
        SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET: PixelPerfectDepth
    }[sharpdepth_kind]

    frozen_denoiser = denoiser_cls.from_pretrained(
        base_ckpt_dir, subfolder=frozen_denoiser_subfolder, revision=None
    )

    vae.requires_grad_(False)
    frozen_denoiser.requires_grad_(False)
    text_encoder.requires_grad_(False)

    student_denoiser = denoiser_cls.from_pretrained(
        student_ckpt_dir, subfolder=denoiser_subfolder, revision=student_ckpt_dir_revision
    )
    student_denoiser.requires_grad_(True)
    
    def disable_dropout(model):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

    # ---------------------------------------------------------------------

    # -------------------- EMA model --------------------
    if args.use_ema:
        ema_model = EMA(
            student_denoiser,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,  # how often to actually update, to save on compute (updates every 10th .update() call)
        )
    # ---------------------------------------------------------------------

    # -------------------- XFORMER --------------------
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            logger.info("enable xformers memory efficient attention")
            frozen_denoiser.enable_xformers_memory_efficient_attention()
            student_denoiser.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    # ---------------------------------------------------------------------

    # -------------------- Gradient checkpointing --------------------
    if args.gradient_checkpointing:
        logger.info("Gradient checkpointing")
        student_denoiser.enable_gradient_checkpointing()  # only student denoiser require grad
    # ---------------------------------------------------------------------

    # -------------------- Sanity check --------------------
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(student_denoiser).dtype != torch.float32:
        raise ValueError(
            f"Student denoiser loaded as datatype {unwrap_model(student_denoiser).dtype}. {low_precision_error_string}"
        )
    if unwrap_model(frozen_denoiser).dtype != torch.float32:
        raise ValueError(
            f"Frozen denoiser loaded as datatype {unwrap_model(frozen_denoiser).dtype}. {low_precision_error_string}"
        )
    # ---------------------------------------------------------------------

    # -------------------- Scale LR --------------------
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    print(f"[process {accelerator.process_index}] real learning rate: {args.learning_rate}")
    # ---------------------------------------------------------------------

    # -------------------- Set up optimizer --------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # let's make a special group for the DiT patch encoder
    params_to_optimize_dit_patch_encoder = [np[1] for np in filter(lambda np: np[1].requires_grad and np[0].startswith("dit.x_embedder.proj"), student_denoiser.named_parameters())]
    other_params_to_optimize = [np[1] for np in filter(lambda np: np[1].requires_grad and not np[0].startswith("dit.x_embedder.proj"), student_denoiser.named_parameters())]

    param_groups = [
        {
            "params": params_to_optimize_dit_patch_encoder,
            "lr": args.learning_rate * args.dit_patch_encoder_lr_multiplier,
        },
        {
            "params": other_params_to_optimize,
            "lr": args.learning_rate,
        }
    ]

    optimizer = optimizer_class(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # ---------------------------------------------------------------------

    # -------------------- Set up training step and LR scheduler --------------------
    # Scheduler and math around the numfprobfber of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps * accelerator.num_processes)) # we multiply by num_processes b/c train_dataloader has not yet been sharded
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )
    # ---------------------------------------------------------------------

    # -------------------- Prepare and move to cuda --------------------
    student_denoiser, frozen_denoiser, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_denoiser, frozen_denoiser, optimizer, train_dataloader, lr_scheduler
    )
    lr_scheduler.split_batches = True

    unwrapped_frozen_denoiser = unwrap_model(frozen_denoiser)
    unwrapped_student_denoiser = unwrap_model(student_denoiser)
    # ---------------------------------------------------------------------

    # -------------------- Set up training precision --------------------
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # ---------------------------------------------------------------------

    # -------------------- Move freezed model to GPU --------------------
    # Move vae, denoiser, and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=weight_dtype)
    frozen_denoiser.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    student_denoiser.to(accelerator.device, dtype=weight_dtype)

    model_architecture = ModelArchitecture(args.base_model)

    def get_checkpoint_from_model_architecture(model_architecture: ModelArchitecture) -> str:
        match model_architecture:
            case ModelArchitecture.unidepth:
                return "lpiccinelli/unidepth-v1-vitl14"
            case ModelArchitecture.depthanythingsmall:
                return "LiheYoung/depth_anything_vits14"
            case ModelArchitecture.depthanythinglarge:
                return "LiheYoung/depth_anything_vitl14"
            case ModelArchitecture.pixelperfectdepth:
                return "andrew-healey/sharpdepth"
            case ModelArchitecture.zoedepth:
                return "isl-org/ZoeDepth"
            case _:
                raise ValueError(f"Invalid model architecture: {model_architecture}")

    base_model_checkpoint = get_checkpoint_from_model_architecture(model_architecture)
    base_depth_estimator_fn = get_depth_estimator_fn(model_architecture, accelerator.device, torch.bfloat16, base_model_checkpoint)


    if args.use_ema:
        ema_model = ema_model.to(accelerator.device, dtype=weight_dtype)
    # ---------------------------------------------------------------------

    # -------------------- Precompute null & task embedding --------------------
    empty_text_emb = encode_empty_text(tokenizer, text_encoder).to(
        accelerator.device, dtype=weight_dtype
    )
    del tokenizer
    del text_encoder
    torch.cuda.empty_cache()
    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    # -------------------- Recalculate training step --------------------
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # ---------------------------------------------------------------------

    # The trackers initializes automatically on the main process.
    # -------------------- Initializes tracker --------------------
    if accelerator.is_main_process:

        init_kwargs = {
            "wandb": {
                "entity": cfg.wandb.entity,
                "name": args.wandb_name
            }
        } if args.report_to == "wandb" else {}

        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)
    # ---------------------------------------------------------------------

    # -------------------------------------------------
    # for reproducibility, log `git diff` + the current commit hash, and throw if it's too big

    if accelerator.is_main_process and args.report_to == "wandb":
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        diff = subprocess.check_output(["git", "diff", commit]).decode("utf-8")

        if len(diff) > 30000: raise ValueError("Git diff is too large, please commit some of your work before training")

        print(f"Commit: {commit}")
        print("<diff>")
        print(diff)
        print("</diff>")

        # upload the diff as an artefact
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(diff.encode("utf-8"))
            f.flush()

            wandb_run = accelerator.get_tracker("wandb",unwrap=True)

            artifact = wandb.Artifact("git-diff", type="diff")
            artifact.add_file(local_path=f.name, name="git-diff.diff")
            wandb_run.log_artifact(artifact)
            print("Uploaded git diff as an artifact")

        print("Json stringified diff:",json.dumps(diff))

        print("Args:")
        pprint(args,expand_all=True)

        print("Command line args:")
        print(" ".join([shlex.quote(arg) for arg in sys.argv]))

        # Print nvidia-smi
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print("nvidia-smi:")
        print(nvidia_smi_output)
    # -------------------------------------------------

    # -------------------- Trainer --------------------
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {sum([len(dataset) for dataset in train_dataset])}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # ---------------------------------------------------------------------

    # -------------------- Resume --------------------
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    # ---------------------------------------------------------------------

    # -------------------- TQDM & stuff --------------------
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )
    device = accelerator.device

    alphas_cumprod = noise_scheduler.alphas_cumprod
    alphas_cumprod = alphas_cumprod.to(device)
    depth_normalizer = ScaleShiftNormalizer()
    # ---------------------------------------------------------------------

    # we keep an EMA of different kinds of losses
    # for logging purposes only!
    # because the loss is not present in all batches
    # and when it is, it might appear more times in one batch than another

    conditioning_kinds = ["no_cond","cond", "all","synthetic"]
    loss_keys = ["total","sds","depth","initial_depth","depth_mse","initial_depth_mse","depth_aligned_mse","final_generation_depth_aligned_mse"]

    # initialize EMA buffers at 0
    loss_exponential_moving_averages = { conditioning_kind: { loss_key: 0.0 for loss_key in loss_keys } for conditioning_kind in conditioning_kinds }
    loss_ema_decay_rate = 0.5 # decays 50% for every training example

    def update_loss_exponential_moving_averages(losses_and_counts):
        for conditioning_kind in conditioning_kinds:
            for loss_key in loss_keys:
                new_loss, new_loss_averaged_over_n_examples = losses_and_counts[conditioning_kind][loss_key]
                old_loss = loss_exponential_moving_averages[conditioning_kind][loss_key]

                new_loss = old_loss * (loss_ema_decay_rate ** new_loss_averaged_over_n_examples) + new_loss * (1 - loss_ema_decay_rate ** new_loss_averaged_over_n_examples)

                loss_exponential_moving_averages[conditioning_kind][loss_key] = new_loss

    loss_accum = torch.tensor(0.0, device=device, dtype=torch.float32)
    last_loss_accum = torch.tensor(0.0, device=device, dtype=torch.float32)

    for epoch in range(first_epoch, args.num_train_epochs):

        curr_train_dataloader = train_dataloader
        if epoch == first_epoch and global_step > epoch * num_update_steps_per_epoch and global_step < (epoch + 1) * num_update_steps_per_epoch:
            curr_train_dataloader = accelerator.skip_first_batches(train_dataloader, (global_step - epoch * num_update_steps_per_epoch) * accelerator.gradient_accumulation_steps)

        student_denoiser.train()
        for step, (batch, row_idx) in enumerate(curr_train_dataloader):
            with accelerator.accumulate(student_denoiser):
                og_batch = batch

                torch.manual_seed(row_idx.item())
                torch.cuda.manual_seed(row_idx.item())
                np.random.seed(row_idx.item())
                random.seed(row_idx.item())

                desired_batch_keys = {"rgb_int", "depth_raw_linear", "valid_mask_raw"}
                desired_batch_keys = {"rgb_int"}
                assert set(batch.keys()) >= desired_batch_keys, f"Invalid batch keys: {set(batch.keys())}. expected it to contain at least these keys: {desired_batch_keys}"
                batch = { key: batch[key] for key in desired_batch_keys }

                # resize the image if using ppd!
                
                rgb = batch["rgb_int"].to(weight_dtype) / 255.0

                assert sharpdepth_kind in [SharpDepthKind.LOTUS, SharpDepthKind.PIXEL_PERFECT_DEPTH, SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET], f"Invalid sharpdepth kind: {sharpdepth_kind}"
                sharpdepth_kind_to_preprocessor = {
                    SharpDepthKind.LOTUS: MarigoldPreProcessor,
                    SharpDepthKind.PIXEL_PERFECT_DEPTH: PixelPerfectDepthPreProcessor,
                    SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET: PixelPerfectDepthPreProcessor,
                }
                preprocessor = sharpdepth_kind_to_preprocessor[sharpdepth_kind]
                rgb_float_1chw_resized, padding, original_resolution = preprocessor.run(batch["rgb_int"], device, weight_dtype)
                rgb_int_1chw_resized = (rgb_float_1chw_resized * 255.0).to(batch["rgb_int"].dtype)
                batch_resized = {
                    "rgb_int": rgb_int_1chw_resized,
                    "depth_raw_linear": None,
                    "valid_mask_raw": None,
                }
                batch = batch_resized
                rgb = rgb_float_1chw_resized
                ## UniDepth ##

                with torch.no_grad():

                    # right now we don't add padding / resize down to the max resolution
                    # weird. this is a train-inference mismatch.
                    # b/c during inference, we *do*!
                    # but it's in the original sharpdepth code, so we'll keep it.
                    # image, _, _ = pipeline.image_processor.preprocess(rgb, 768, "bilinear", accelerator.device)  # [N,3,PPH,PPW]

                    disp_base = disp_base_11hw = base_depth_estimator_fn(og_batch["rgb_int"], preprocessor,internal=True)
                    assert disp_base_11hw.shape[2:] == rgb.shape[2:], f"Base depth map doesn't match its input image resolution! disp_base_11hw.shape[2:] = {disp_base_11hw.shape[2:]}, rgb_float_1chw.shape[2:] = {rgb.shape[2:]}"

                normalize_obj = depth_normalizer(disp_base)
                norm_base_depth = normalize_obj["norm_depth"].to(dtype=weight_dtype)
                norm_base_depth_unpadded = norm_base_depth[:,:,:norm_base_depth.shape[2]-padding[0],:norm_base_depth.shape[3]-padding[1]]
 
                if sharpdepth_kind == SharpDepthKind.LOTUS:

                    assert args.blur_unidepth_output_ratio == 1, "Blurring the unidepth output is not supported for Lotus"

                    # 1. Encode depth (totally lotus-specific)
                    unidepth_latent = encode_depth(vae, norm_base_depth)

                    rgb = rgb * 2 - 1

                    ## Lotus ##
                    # Encode image, text, and timestep
                    # for PPD, we'd have to preprocess timestep and resize the image properly
                    with torch.no_grad():
                        rgb_latent = encode_image(vae, rgb)
                        lotus_timesteps = torch.ones((rgb_latent.shape[0],), device=device) * (
                            noise_scheduler.config.num_train_timesteps - 1
                        )
                        lotus_timesteps = lotus_timesteps.long()
                        batch_empty_text_embed = empty_text_emb.repeat((rgb_latent.shape[0], 1, 1)).to(
                            device, dtype=weight_dtype
                        )
                        batch_task_emb = task_emb.repeat((rgb_latent.shape[0], 1)).to(
                            device, dtype=weight_dtype
                        )

                    # ---------------------------------
                    # extract mask
                    # tbh we'd just wrap this whole thing in an if/else statement
                    with torch.no_grad():
                        lotus_input = torch.cat(
                            [rgb_latent.detach(), torch.randn_like(rgb_latent)], dim=1
                        )  # this order is important

                        if args.use_ema:
                            lotus_pred = ema_model(
                                lotus_input,
                                lotus_timesteps.to(weight_dtype),
                                batch_empty_text_embed,
                                class_labels=batch_task_emb,
                            ).sample
                        else:
                            lotus_pred = student_denoiser(
                                lotus_input,
                                lotus_timesteps.to(weight_dtype),
                                batch_empty_text_embed,
                                class_labels=batch_task_emb,
                            ).sample

                        # ---------------------------------
                        # decode pred_latent to depth
                        latent = lotus_pred / vae.config.scaling_factor
                        z = vae.post_quant_conv(latent.to(weight_dtype))
                        frozen_pred_depth = vae.decoder(z).mean(dim=1, keepdim=True)

                        frozen_pred_depth_unpadded = frozen_pred_depth[:,:,:frozen_pred_depth.shape[2]-padding[0],:frozen_pred_depth.shape[3]-padding[1]]

                        # ---------------------------------
                        # calculate difference
                        l1_error = torch.abs(frozen_pred_depth - norm_base_depth)
                        l1_error = l1_error / l1_error.max()
                        l1_error = l1_error.clip(0, 1)

                        l1_error_unpadded = l1_error[:,:,:l1_error.shape[2]-padding[0],:l1_error.shape[3]-padding[1]]

                        latent_mask = torch.nn.functional.interpolate(l1_error, scale_factor=1 / 8)

                        noise = torch.randn_like(unidepth_latent)
                        noisy_lotus_latent = noise_scheduler.add_noise(
                            lotus_pred, noise, lotus_timesteps
                        )

                        # ah ok, a simple way to mitigate catastrophic forgetting. makes sense, I think.
                        if torch.rand(1).item() < args.use_conditioning_probability:
                            conditioning_kind = "cond"
                            noisy_latent = noisy_lotus_latent * latent_mask + unidepth_latent * (
                                1 - latent_mask
                            )
                            student_input = torch.cat([rgb_latent, noisy_latent], dim=1)
                        else:
                            conditioning_kind = "no_cond"
                            student_input = torch.cat([rgb_latent, noise], dim=1)

                    # ---------------------------------
                    pred_latent = student_denoiser(
                        student_input,
                        lotus_timesteps.to(weight_dtype),
                        encoder_hidden_states=batch_empty_text_embed,
                        class_labels=batch_task_emb,
                    ).sample

                    # ------------------------------------------------------------------
                    # SDS loss
                    noise = torch.randn_like(pred_latent)
                    noisy_samples = noise_scheduler.add_noise(pred_latent, noise, lotus_timesteps)
                    denoiser_input = torch.cat([rgb_latent.detach(), noisy_samples], dim=1).to(
                        weight_dtype
                    )  # this order is important

                    with torch.no_grad():
                        denoiser_pred = frozen_denoiser(
                            denoiser_input,
                            lotus_timesteps.to(weight_dtype),
                            batch_empty_text_embed,
                            class_labels=batch_task_emb,
                        ).sample

                    sigma_t = ((1 - alphas_cumprod[lotus_timesteps]) ** 0.5).view(-1, 1, 1, 1)
                    score_gradient = torch.nan_to_num(sigma_t**2 * (pred_latent - denoiser_pred))
                    # ------------------------------------------------------------------

                    # ------------------------------------------------------------------
                    # Compute the SDS loss for the model
                    target = (pred_latent - score_gradient).detach()
                    sds_loss = 0.5 * F.mse_loss(pred_latent.float(), target.float(), reduction="mean")
                    # ------------------------------------------------------------------

                    # ---------------------------------
                    # decode pred_latent to depth
                    latent = pred_latent / vae.config.scaling_factor
                    z = vae.post_quant_conv(latent.to(weight_dtype))
                    pred_depth = vae.decoder(z).mean(dim=1, keepdim=True)

                    pred_depth_unpadded = pred_depth[:,:,:pred_depth.shape[2]-padding[0],:pred_depth.shape[3]-padding[1]]

                    depth_loss = l1_loss(
                        pred_depth_unpadded * 0.5 + 0.5, norm_base_depth_unpadded * 0.5 + 0.5, l1_error_unpadded
                    )
                    depth_mse = F.mse_loss(pred_depth_unpadded * 0.5 + 0.5, norm_base_depth_unpadded * 0.5 + 0.5, reduction="mean") 

                    # let's also compare our final predicted depth map to a simple baseline: least-squares alignment with the base depth map

                    initial_depth_loss = None
                    if torch.rand(1).item() < args.compute_initial_depth_loss_probability:
                        with torch.no_grad():

                            frozen_pred_depth_aligned, _, _ = align_depth_least_square(
                                gt_arr=(norm_base_depth_unpadded * 0.5 + 0.5).detach().float().cpu().numpy(),
                                pred_arr=frozen_pred_depth_unpadded.detach().float().cpu().numpy(),
                                valid_mask_arr=torch.ones_like(l1_error_unpadded).detach().bool().cpu().numpy(),
                                return_scale_shift=True,
                                max_resolution=None,
                            )
                            frozen_pred_depth_aligned = torch.from_numpy(frozen_pred_depth_aligned).to(device)

                            initial_depth_loss = l1_loss(
                                frozen_pred_depth_aligned, norm_base_depth_unpadded * 0.5 + 0.5, l1_error_unpadded
                            )
                            initial_depth_mse = F.mse_loss(frozen_pred_depth_aligned, norm_base_depth_unpadded * 0.5 + 0.5, reduction="mean")
                    
                    with torch.no_grad():
                        final_aligned_depth, _, _ = align_depth_least_square(
                            gt_arr=(norm_base_depth_unpadded * 0.5 + 0.5).detach().float().cpu().numpy(),
                            pred_arr=pred_depth_unpadded.detach().float().cpu().numpy(),
                            valid_mask_arr=torch.ones_like(l1_error_unpadded).detach().bool().cpu().numpy(),
                            return_scale_shift=True,
                            max_resolution=None,
                        )
                        final_aligned_depth = torch.from_numpy(final_aligned_depth).to(device)

                        if args.use_normal_loss:
                            intrinsics = og_batch["intrinsics"].to(device)

                            final_aligned_metric_depth, _, _ = align_depth_least_square(
                                gt_arr=(disp_base_11hw).detach().float().cpu().numpy(),
                                pred_arr=pred_depth.detach().float().cpu().numpy(),
                                valid_mask_arr=torch.ones_like(l1_error_unpadded).detach().bool().cpu().numpy(),
                                return_scale_shift=True,
                                max_resolution=None,
                            )
                            final_aligned_metric_depth = torch.from_numpy(final_aligned_metric_depth).to(device)

                            base_normal_map = depth_to_normals(disp_base_11hw.to(device), intrinsics)
                            pred_normal_map = depth_to_normals(final_aligned_metric_depth, intrinsics)
                            normal_loss = l1_loss(base_normal_map, pred_normal_map, torch.ones(pred_normal_map.shape, dtype=pred_normal_map.dtype, device=pred_normal_map.device))


                        final_aligned_depth_loss = l1_loss(
                            final_aligned_depth, norm_base_depth_unpadded * 0.5 + 0.5, l1_error_unpadded
                        )
                        final_aligned_depth_mse = F.mse_loss(final_aligned_depth, norm_base_depth_unpadded * 0.5 + 0.5, reduction="mean")

                        final_generation_aligned_depth_mse = final_aligned_depth_mse
                        final_generation_aligned_depth_loss = final_aligned_depth_loss

                    if accelerator.is_main_process and args.log_depth_maps and step % 10 == 0:
                        with torch.no_grad():

                            os.makedirs("/tmp/viz", exist_ok=True)

                            rgb_viz = (rgb + 1) / 2

                            def colorize_internal(value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"):
                                colored = colorize_depth_maps(value.squeeze(0), vmin, vmax, cmap)
                                colored = (colored * 255).astype(np.uint8)
                                colored_hwc = chw2hwc(colored.squeeze(0))
                                return Image.fromarray(colored_hwc)
                            
                            score_gradient_latent = score_gradient / vae.config.scaling_factor
                            score_gradient_z = vae.post_quant_conv(score_gradient_latent.to(weight_dtype))
                            score_gradient_depth = vae.decoder(score_gradient_z).mean(dim=1, keepdim=True)
                            sds_score = score_gradient_depth.abs().float()
                            sds_score_colored = colorize_internal(sds_score.cpu().numpy(), sds_score.min().item(), sds_score.max().item(), cmap="coolwarm")
                            sds_score_colored.save("/tmp/viz/sds_score.png")

                            l1_error_colored = colorize_internal(l1_error.float().cpu().numpy(), 0, 1, cmap="coolwarm")
                            l1_error_colored.save("/tmp/viz/l1_error.png")

                            weighted_sds_score = (sds_score * ((1 - l1_error.float())**2)).float()
                            weighted_sds_score_colored = colorize_internal(weighted_sds_score.cpu().numpy(), weighted_sds_score.min().item(), weighted_sds_score.max().item(), cmap="Greys")
                            weighted_sds_score_colored.save("/tmp/viz/weighted_sds_score.png")

                            base_depth_colored = colorize_internal(norm_base_depth.float().cpu().numpy(), norm_base_depth.float().min().item(), norm_base_depth.float().max().item(), cmap="coolwarm")
                            base_depth_colored.save("/tmp/viz/base_depth.png")

                            initial_depth_colored = colorize_internal(frozen_pred_depth.float().cpu().numpy(), frozen_pred_depth.float().min().item(), frozen_pred_depth.float().max().item(), cmap="coolwarm")
                            initial_depth_colored.save("/tmp/viz/initial_depth.png")

                            final_depth_colored = colorize_internal(pred_depth.float().cpu().numpy(), pred_depth.float().min().item(), pred_depth.float().max().item(), cmap="coolwarm")
                            final_depth_colored.save("/tmp/viz/final_depth.png")

                            denoiser_pred_latent = denoiser_pred / vae.config.scaling_factor
                            denoiser_pred_z = vae.post_quant_conv(denoiser_pred_latent.to(weight_dtype))
                            frozen_denoiser_pred_depth = vae.decoder(denoiser_pred_z).mean(dim=1, keepdim=True).float()
                            frozen_denoiser_pred_depth_colored = colorize_internal(frozen_denoiser_pred_depth.cpu().numpy(), frozen_denoiser_pred_depth.min().item(), frozen_denoiser_pred_depth.max().item(), cmap="coolwarm")
                            frozen_denoiser_pred_depth_colored.save("/tmp/viz/frozen_denoiser_pred_depth.png")

                            rgb_img = Image.fromarray(((rgb_viz.float() * 255.0).int().squeeze(0,1).permute(1,2,0).cpu().numpy().astype(np.uint8)))

                            # let's concatenate them vertically!
                            concatenated = np.concatenate([np.array(img) for img in [
                                sds_score_colored,
                                l1_error_colored,
                                weighted_sds_score_colored,
                                base_depth_colored,
                                initial_depth_colored,
                                final_depth_colored,
                                frozen_denoiser_pred_depth_colored,
                                rgb_img,
                            ]],axis=0)
                            Image.fromarray(concatenated).save("/tmp/viz/concatenated.png")

                            pass

                
                elif sharpdepth_kind == SharpDepthKind.PIXEL_PERFECT_DEPTH:

                    if args.use_normal_loss:
                        raise NotImplementedError("Normal consistency loss not supported for ppd sharpdepth.")

                    norm_base_depth = norm_base_depth * 0.5 + 0.5

                    # ---------------------------------
                    # calculate difference

                    def blur(x_11hw, scale_factor):
                        if args.gaussian_blur:
                            return TV_F.gaussian_blur(x_11hw, kernel_size=2*(scale_factor//2)+1, sigma=scale_factor/2)
                        else:
                            small_h = x_11hw.shape[2] // scale_factor
                            small_w = x_11hw.shape[3] // scale_factor
                            downscaled = F.interpolate(x_11hw, size=(small_h, small_w), mode="area")
                            upscaled = F.interpolate(downscaled, size=(x_11hw.shape[2], x_11hw.shape[3]), mode="bilinear")
                            return upscaled
                    
                    def maybe_blur(x_11hw):
                        if args.blur_unidepth_output_ratio != 1:
                            return blur(x_11hw, args.blur_unidepth_output_ratio)
                        else:
                            return x_11hw
                            
                    # initial PPD
                    with torch.no_grad():
                        cond = rgb - 0.5
                        noise = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(device)

                        if args.initialize_ppd_from_timestep is not None and forward_diffuse_from == ForwardDiffuseFrom.INITIAL_PRED_DEPTH:
                            timesteps = torch.tensor([timestep for timestep in unwrapped_student_denoiser.sampling_timesteps if timestep <= args.initialize_ppd_from_timestep],device=device,dtype=weight_dtype)
                            latent = unwrapped_student_denoiser.schedule.forward(norm_base_depth - 0.5, noise, torch.tensor(args.initialize_ppd_from_timestep,device=device,dtype=weight_dtype))
                        else:
                            timesteps = torch.tensor(unwrapped_student_denoiser.sampling_timesteps,device=device,dtype=weight_dtype)
                            latent = noise

                        if forward_diffuse_from == ForwardDiffuseFrom.INITIAL_PRED_DEPTH:
                            with torch.autocast(device.type,dtype=weight_dtype):
                                semantics = unwrapped_frozen_denoiser.semantics_prompt(rgb)
                                noisy_depth_cond = (norm_base_depth - 0.5) if args.use_conditioning_for_initial_ppd else torch.randn_like(latent)
                                for timestep in timesteps:
                                    input = torch.cat([latent, cond, noisy_depth_cond], dim=1)
                                    pred = student_denoiser(x=input, semantics=semantics, timestep=timestep)
                                    latent = unwrapped_student_denoiser.sampler.step(pred=pred, x_t=latent, t=timestep)
                                frozen_pred_depth = latent + 0.5
                        else:
                            with torch.autocast(device.type,dtype=weight_dtype):
                                semantics = unwrapped_frozen_denoiser.semantics_prompt(rgb)
                                for timestep in timesteps:
                                    input = torch.cat([latent, cond], dim=1)
                                    pred = frozen_denoiser(x=input, semantics=semantics, timestep=timestep)
                                    latent = unwrapped_frozen_denoiser.sampler.step(pred=pred, x_t=latent, t=timestep)
                                frozen_pred_depth = latent + 0.5
                    
                        l1_error = torch.abs(frozen_pred_depth - norm_base_depth)
                        l1_error = l1_error / l1_error.max()
                        l1_error = l1_error.clip(0, 1)
                        l1_mask = l1_error
                        
                        timestep = timesteps[random.randrange(0, len(timesteps))]

                        should_use_conditioning = torch.rand(1).item() < args.use_conditioning_probability
                        if should_use_conditioning:
                            conditioning_kind = "cond"
                            target_latent = norm_base_depth - 0.5

                            if forward_diffuse_from == ForwardDiffuseFrom.INITIAL_PRED_DEPTH:
                                x0 = frozen_pred_depth - 0.5
                            elif forward_diffuse_from == ForwardDiffuseFrom.BASE_PRED_DEPTH:
                                x0 = norm_base_depth - 0.5
                            else:
                                raise ValueError(f"Invalid forward diffuse from: {forward_diffuse_from}")
                            xT = torch.randn_like(latent)
                            xt = unwrapped_student_denoiser.schedule.forward(x0, xT, timestep)
                            
                            scaled_l1_mask = l1_mask * args.noise_aware_latent_noise_scale
                            noisy_depth_cond = (norm_base_depth - 0.5) * (1 - scaled_l1_mask) + (torch.randn_like(x0) * scaled_l1_mask)

                            student_input = torch.cat([xt, cond, noisy_depth_cond], dim=1)
                        else:
                            conditioning_kind = "no_cond"
                            target_latent = norm_base_depth - 0.5

                            x0 = norm_base_depth - 0.5
                            xT = noise
                            xt = unwrapped_student_denoiser.schedule.forward(x0, xT, timestep)

                            noisy_depth_cond = torch.randn_like(xt)

                            student_input = torch.cat([xt, cond, noisy_depth_cond], dim=1)
                    
                    with torch.autocast(device.type,dtype=weight_dtype):
                        student_pred_depth_latent_velocity = student_denoiser(x=student_input, semantics=semantics, timestep=timestep)
                        student_pred_depth_latent, _ = unwrapped_student_denoiser.schedule.convert_from_pred(student_pred_depth_latent_velocity, 'velocity', xt, timestep)
                        pred_depth = student_pred_depth = student_pred_depth_latent + 0.5

                        available_sds_timesteps = [timestep for timestep in unwrapped_student_denoiser.sampling_timesteps if timestep <= args.max_sds_timestep]
                        sds_timestep = torch.tensor(random.choice(available_sds_timesteps), device=device, dtype=weight_dtype)

                        # SDS loss
                        noise = torch.randn_like(student_pred_depth_latent)
                        noised_student_pred_depth_latent = unwrapped_student_denoiser.schedule.forward(student_pred_depth_latent, noise, sds_timestep)
                        with torch.no_grad():
                            frozen_denoiser_input = torch.cat([noised_student_pred_depth_latent, cond], dim=1)
                            frozen_denoiser_pred_depth_latent_velocity = frozen_denoiser(frozen_denoiser_input, semantics=semantics, timestep=sds_timestep)
                            frozen_denoiser_pred_depth_latent, pred_noise = unwrapped_frozen_denoiser.schedule.convert_from_pred(frozen_denoiser_pred_depth_latent_velocity, 'velocity', noised_student_pred_depth_latent, sds_timestep)
                            frozen_denoiser_pred_depth = frozen_denoiser_pred_depth_latent + 0.5

                            # TODO figure out if the sign should be reversed here
                            score_vector = (pred_noise.float() - noise.float())

                    # let's do sds loss only in the regions with low l1 error 

                    if args.use_edge_loss_as_sds_loss:
                        assert forward_diffuse_from == ForwardDiffuseFrom.INITIAL_PRED_DEPTH, "Edge loss as SDS loss is only supported for initial pred depth, b/c we can only use the frozen_pred_depth edges as a pseudo-ground truth when it's the real ppd base depth"

                        high_frequency_frozen_pred_depth = frozen_pred_depth - maybe_blur(frozen_pred_depth)
                        high_frequency_student_pred_depth = student_pred_depth_latent - maybe_blur(student_pred_depth_latent)
                        
                        is_near_edge_mask = get_dilated_edge_mask(frozen_pred_depth, distance_threshold_px=args.edge_loss_blur_radius_px)
                        edge_loss = F.mse_loss(high_frequency_frozen_pred_depth * is_near_edge_mask, high_frequency_student_pred_depth * is_near_edge_mask, reduction="mean").to(weight_dtype) / (is_near_edge_mask.float().mean() + 1e-6)
                        sds_loss = edge_loss

                    else:

                        high_freq_sds_score_vector = score_vector#(score_vector - maybe_blur(score_vector)).abs() * ((1-maybe_blur(l1_error))**2)
                        sds_loss = 0.5 * F.mse_loss(student_pred_depth_latent.float(), (student_pred_depth_latent.float() - high_freq_sds_score_vector.float()).detach().float(), reduction="mean")

                    if accelerator.is_main_process and args.log_depth_maps and step % 10 == 0:
                        with torch.no_grad():

                            os.makedirs("/tmp/viz", exist_ok=True)

                            Image.fromarray(((rgb * 255.0).int().squeeze(0,1).permute(1,2,0).cpu().numpy().astype(np.uint8))).save("/tmp/viz_rgb.png")

                            def colorize_internal(value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"):
                                colored = colorize_depth_maps(value.squeeze(0), vmin, vmax, cmap)
                                colored = (colored * 255).astype(np.uint8)
                                colored_hwc = chw2hwc(colored.squeeze(0))
                                return Image.fromarray(colored_hwc)
                            
                            
                            sds_score = score_vector.abs()
                            sds_score_colored = colorize_internal(sds_score.cpu().numpy(), sds_score.min().item(), sds_score.max().item(), cmap="coolwarm")
                            sds_score_colored.save("/tmp/viz/sds_score.png")

                            l1_error_colored = colorize_internal(maybe_blur(l1_error).cpu().numpy(), 0, 1, cmap="coolwarm")
                            l1_error_colored.save("/tmp/viz/l1_error.png")
                            # print("l1_error.min(), l1_error.max()", l1_error.min().item(), l1_error.max().item())

                            weighted_sds_score = (sds_score - maybe_blur(sds_score)).abs() * ((1-maybe_blur(l1_error))**2)
                            weighted_sds_score_colored = colorize_internal(weighted_sds_score.cpu().numpy(), weighted_sds_score.min().item(), weighted_sds_score.max().item(), cmap="Greys")
                            weighted_sds_score_colored.save("/tmp/viz/weighted_sds_score.png")

                            base_depth_colored = colorize_internal(norm_base_depth.float().cpu().numpy(), norm_base_depth.min().item(), norm_base_depth.max().item(), cmap="coolwarm")
                            base_depth_colored.save("/tmp/viz/base_depth.png")

                            initial_depth_colored = colorize_internal(frozen_pred_depth.cpu().numpy(), frozen_pred_depth.min().item(), frozen_pred_depth.max().item(), cmap="coolwarm")
                            initial_depth_colored.save("/tmp/viz/initial_depth.png")

                            final_depth_colored = colorize_internal(student_pred_depth.cpu().numpy(), student_pred_depth.min().item(), student_pred_depth.max().item(), cmap="coolwarm")
                            final_depth_colored.save("/tmp/viz/final_depth.png")

                            frozen_denoiser_pred_depth_colored = colorize_internal(frozen_denoiser_pred_depth.cpu().numpy(), frozen_denoiser_pred_depth.min().item(), frozen_denoiser_pred_depth.max().item(), cmap="coolwarm")
                            frozen_denoiser_pred_depth_colored.save("/tmp/viz/frozen_denoiser_pred_depth.png")

                            # sds_input_depth = colorize_internal()

                            rgb_img = Image.fromarray(((rgb * 255.0).int().squeeze(0,1).permute(1,2,0).cpu().numpy().astype(np.uint8)))

                            # let's concatenate them vertically!
                            concatenated = np.concatenate([np.array(img) for img in [
                                sds_score_colored,
                                l1_error_colored,
                                weighted_sds_score_colored,
                                base_depth_colored,
                                initial_depth_colored,
                                final_depth_colored,
                                frozen_denoiser_pred_depth_colored,
                                rgb_img,
                            ]],axis=0)
                            Image.fromarray(concatenated).save("/tmp/viz/concatenated.png")

                            pass
                            
                    maybe_blur_depth_loss = lambda x: maybe_blur(x) if args.blur_depth_loss else x

                    if args.use_edge_loss_as_sds_loss:
                        is_far_from_edges_mask = torch.logical_not(get_dilated_edge_mask(frozen_pred_depth, distance_threshold_px=args.depth_loss_away_from_edges_threshold_px))
                        assert is_far_from_edges_mask.shape == student_pred_depth.shape

                        depth_loss = l1_loss(
                            maybe_blur_depth_loss(student_pred_depth_latent + 0.5) * is_far_from_edges_mask, maybe_blur_depth_loss(target_latent + 0.5) * is_far_from_edges_mask, is_far_from_edges_mask.to(torch.bool)
                        )

                    else:
                        depth_loss = l1_loss(
                            maybe_blur_depth_loss(student_pred_depth_latent + 0.5), maybe_blur_depth_loss(target_latent + 0.5), torch.ones_like(l1_error)
                        ) 

                    depth_mse = F.mse_loss(student_pred_depth_latent + 0.5, target_latent + 0.5, reduction="mean")

                    with torch.no_grad():

                        # let's do multiple diffusion steps
                        if args.initialize_ppd_from_timestep is not None:
                            timesteps = torch.tensor([timestep for timestep in unwrapped_student_denoiser.sampling_timesteps if timestep <= args.initialize_ppd_from_timestep],device=device,dtype=weight_dtype)
                            final_generation_latent = unwrapped_student_denoiser.schedule.forward(student_pred_depth_latent, noise, torch.tensor(args.initialize_ppd_from_timestep,device=device,dtype=weight_dtype))
                        else:
                            timesteps = torch.tensor(unwrapped_student_denoiser.sampling_timesteps,device=device,dtype=weight_dtype)
                            final_generation_latent = noise

                        for timestep in timesteps:
                            input = torch.cat([final_generation_latent, cond, noisy_depth_cond], dim=1)
                            pred = student_denoiser(x=input, semantics=semantics, timestep=timestep)
                            final_generation_latent = unwrapped_student_denoiser.sampler.step(pred=pred, x_t=final_generation_latent, t=timestep)
                        final_generation_depth = final_generation_latent + 0.5

                        final_pred_depth_aligned, _, _ = align_depth_least_square(
                            gt_arr=(target_latent + 0.5).detach().float().cpu().numpy(),
                            pred_arr=student_pred_depth.detach().float().cpu().numpy(),
                            valid_mask_arr=torch.ones_like(l1_error).detach().bool().cpu().numpy(),
                            return_scale_shift=True,
                            max_resolution=None,
                        )
                        final_pred_depth_aligned = torch.from_numpy(final_pred_depth_aligned).to(device)
                        final_aligned_depth_mse = F.mse_loss(final_pred_depth_aligned, target_latent + 0.5, reduction="mean")

                        final_generation_depth_aligned, _, _ = align_depth_least_square(
                            gt_arr=(target_latent + 0.5).detach().float().cpu().numpy(),
                            pred_arr=final_generation_depth.detach().float().cpu().numpy(),
                            valid_mask_arr=torch.ones_like(l1_error).detach().bool().cpu().numpy(),
                            return_scale_shift=True,
                            max_resolution=None,
                        )
                        final_generation_depth_aligned = torch.from_numpy(final_generation_depth_aligned).to(device)
                        final_generation_aligned_depth_mse = F.mse_loss(final_generation_depth_aligned, target_latent + 0.5, reduction="mean")

                    # let's also compare our final predicted depth map to a simple baseline: least-squares alignment with the base depth map

                    initial_depth_loss = None
                    if torch.rand(1).item() < args.compute_initial_depth_loss_probability:
                        with torch.no_grad():

                            frozen_pred_depth_aligned, _, _ = align_depth_least_square(
                                gt_arr=(target_latent + 0.5).detach().float().cpu().numpy(),
                                pred_arr=frozen_pred_depth.detach().float().cpu().numpy(),
                                valid_mask_arr=torch.ones_like(l1_error).detach().bool().cpu().numpy(),
                                return_scale_shift=True,
                                max_resolution=None,
                            )
                            frozen_pred_depth_aligned = torch.from_numpy(frozen_pred_depth_aligned).to(device)

                            initial_depth_loss = l1_loss(
                                maybe_blur_depth_loss(frozen_pred_depth_aligned), maybe_blur_depth_loss(target_latent + 0.5), torch.ones_like(l1_error)
                            )
                            initial_depth_mse = F.mse_loss(frozen_pred_depth_aligned, target_latent + 0.5, reduction="mean")

                elif sharpdepth_kind == SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET:
                    if args.use_normal_loss:
                        raise NotImplementedError("Normal consistency loss not supported for ppd sharpdepth.")
                    # infer the frozen ppd on the image
                    # apply some arb. rescaling to the ppd depth map, e.g. linear or log(linear(exp() + 1)) - 1
                    # and blur it
                    # then perform simple diffusion towards the rescaled depth map!

                    norm_base_depth = norm_base_depth * 0.5 + 0.5 # normalize to [0, 1]

                    def blur(x_11hw, scale_factor):
                        if args.gaussian_blur:
                            return TV_F.gaussian_blur(x_11hw, kernel_size=2*(scale_factor//2)+1, sigma=scale_factor/2)
                        else:
                            small_h = x_11hw.shape[2] // scale_factor
                            small_w = x_11hw.shape[3] // scale_factor
                            downscaled = F.interpolate(x_11hw, size=(small_h, small_w), mode="area")
                            upscaled = F.interpolate(downscaled, size=(x_11hw.shape[2], x_11hw.shape[3]), mode="bilinear")
                            return upscaled
                    
                    def maybe_blur(x_11hw):
                        if args.blur_unidepth_output_ratio != 1:
                            return blur(x_11hw, args.blur_unidepth_output_ratio)
                        else:
                            return x_11hw

                    # infer ppd
                    with torch.no_grad():
                        cond = rgb - 0.5
                        noise = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(device)

                        timesteps = unwrapped_student_denoiser.sampling_timesteps
                        latent = noise

                        with torch.autocast(device.type,dtype=weight_dtype):
                            semantics = unwrapped_frozen_denoiser.semantics_prompt(rgb)
                            for timestep in timesteps:
                                input = torch.cat([latent, cond], dim=1)
                                pred = unwrapped_frozen_denoiser(x=input, semantics=semantics, timestep=timestep)
                                latent = unwrapped_student_denoiser.sampler.step(pred=pred, x_t=latent, t=timestep)
                            frozen_pred_depth = latent + 0.5

                    if random.random() < args.use_synthetic_conditioning_probability:
                        conditioning_kind = "synthetic"
                    
                        # now transform it!
                        transformation_kinds = ["log_rescale"]
                        transformation_kind = random.choice(transformation_kinds)
                        if transformation_kind == "log_rescale":
                            rand_rescale_factor = math.exp(random.uniform(math.log(0.1), math.log(10.0))) # uniform in log space
                            # ppd operates in log-space. normalized ppd latent = log(depth + 1). to avoid discontinuities at depth~=zero.
                            rescaled_frozen_pred_depth = torch.log((torch.exp(frozen_pred_depth) - 1) * rand_rescale_factor + 1)
                            
                            rescaled_normalize_obj = depth_normalizer(rescaled_frozen_pred_depth)
                            norm_rescaled_frozen_pred_depth = rescaled_normalize_obj["norm_depth"] * 0.5 + 0.5

                            rand_rescale_factor = math.exp(random.uniform(math.log(0.1), math.log(1.0))) # uniform in log space
                            rand_sign_change = random.choice([1, -1]) if args.flip_sign_for_controlnet else 1
                            rand_bias = random.uniform(-0.1, 0.1)
                            norm_rescaled_frozen_pred_depth = norm_rescaled_frozen_pred_depth * rand_rescale_factor * rand_sign_change + rand_bias

                        else:
                            raise NotImplementedError(f"Unknown transformation kind: {transformation_kind}")
                        
                        # now blur it!
                        blurred_norm_rescaled_frozen_pred_depth = maybe_blur(norm_rescaled_frozen_pred_depth)

                        target_depth = norm_rescaled_frozen_pred_depth
                        blurred_target_depth = blurred_norm_rescaled_frozen_pred_depth
                    else:
                        conditioning_kind = "cond"
                        target_depth = norm_base_depth
                        blurred_target_depth = maybe_blur(norm_base_depth)
                    
                    # now perform simple diffusion towards the rescaled depth map!
                    timestep = unwrapped_student_denoiser.sampling_timesteps[random.randrange(0, len(unwrapped_student_denoiser.sampling_timesteps))]

                    x0 = None

                    forward_diffuse_from = None
                    if args.forward_diffuse_from == "initial_pred_depth":
                        if random.random() < args.forward_diffuse_from_initial_pred_depth_probability:
                            forward_diffuse_from = "initial_pred_depth"
                        else:
                            forward_diffuse_from = "base_pred_depth"
                    elif args.forward_diffuse_from == "base_pred_depth":
                        forward_diffuse_from = "base_pred_depth"
                    else:
                        raise ValueError(f"Unknown 'forward diffuse from' setting: {args.forward_diffuse_from}")

                    if forward_diffuse_from == "initial_pred_depth":
                        with torch.no_grad():

                            noise = torch.randn_like(target_depth)
                            latent = noise
                            cond_inputs = [
                                torch.cat([target_depth - 0.5, cond], dim=1),
                                torch.cat([frozen_pred_depth - 0.5, cond], dim=1),
                            ]
                            for loop_timestep in unwrapped_student_denoiser.sampling_timesteps:
                                student_dit_input = torch.cat([latent, cond], dim=1)
                                pred_velocity = student_denoiser(x=student_dit_input, conds=cond_inputs, semantics=semantics, timestep=loop_timestep)
                                latent = unwrapped_student_denoiser.sampler.step(pred=pred_velocity, x_t=latent, t=loop_timestep)
                            x0 = latent
                    elif forward_diffuse_from == "base_pred_depth":
                        x0 = target_depth - 0.5
                    else:
                        raise ValueError(f"Unknown 'forward diffuse from' setting: {args.forward_diffuse_from}")

                    xT = torch.randn_like(noise)
                    xt = unwrapped_student_denoiser.schedule.forward(x0, xT, timestep)

                    student_dit_input = torch.cat([xt, cond], dim=1)
                    cond_inputs = [
                        torch.cat([target_depth - 0.5, cond], dim=1),
                        torch.cat([frozen_pred_depth - 0.5, cond], dim=1),
                    ]

                    with torch.autocast(device.type,dtype=weight_dtype):
                        student_pred_depth_latent_velocity = student_denoiser(x=student_dit_input, conds=cond_inputs, semantics=semantics, timestep=timestep)
                        student_pred_depth_latent, _ = unwrapped_student_denoiser.schedule.convert_from_pred(student_pred_depth_latent_velocity, 'velocity', xt, timestep)
                        student_pred_depth = student_pred_depth_latent + 0.5
                    
                    is_far_from_edges_mask = torch.logical_not(get_dilated_edge_mask(frozen_pred_depth, distance_threshold_px=args.depth_loss_away_from_edges_threshold_px))
                    assert is_far_from_edges_mask.shape == student_pred_depth.shape

                    # let's just use an edge loss as our sds loss.
                    high_frequency_initial_depth = frozen_pred_depth - blur(frozen_pred_depth, args.edge_loss_blur_radius_px)
                    high_frequency_student_pred_depth = student_pred_depth - blur(student_pred_depth, args.edge_loss_blur_radius_px)
                    is_near_edge_mask = get_dilated_edge_mask(student_pred_depth, distance_threshold_px=args.edge_loss_blur_radius_px)
 
                    if conditioning_kind == "synthetic":
                        depth_loss = F.mse_loss(student_pred_depth, target_depth, reduction="mean")
                        edge_loss = F.mse_loss(high_frequency_initial_depth * is_near_edge_mask, high_frequency_student_pred_depth * is_near_edge_mask, reduction="mean").to(weight_dtype) / (is_near_edge_mask.float().mean() + 1e-6)
                        sds_loss = edge_loss
 
                    elif conditioning_kind == "cond":

                        # sharpdepth-style losses! l1-weighted depth map and sds loss
                        if args.use_sharpdepth_style_losses:
                            l1_error = torch.abs(frozen_pred_depth - target_depth)
                            depth_loss = l1_loss(
                                student_pred_depth, target_depth, l1_error
                            )

                            # now compute sds loss!
                            available_sds_timesteps = [timestep for timestep in unwrapped_student_denoiser.sampling_timesteps if timestep <= args.max_sds_timestep]
                            assert len(available_sds_timesteps) == 1, "We only support one SDS timestep for sharpdepth style losses"
                            sds_timestep = torch.tensor(random.choice(available_sds_timesteps), device=device, dtype=weight_dtype)

                            noise = torch.randn_like(student_pred_depth_latent)
                            noised_student_pred_depth_latent = unwrapped_student_denoiser.schedule.forward(student_pred_depth_latent, noise, sds_timestep)
                            with torch.no_grad():
                                frozen_denoiser_input = torch.cat([noised_student_pred_depth_latent, cond], dim=1)
                                frozen_denoiser_pred_depth_latent_velocity = frozen_denoiser(frozen_denoiser_input, semantics=semantics, timestep=sds_timestep)
                                frozen_denoiser_pred_depth_latent, pred_noise = unwrapped_frozen_denoiser.schedule.convert_from_pred(frozen_denoiser_pred_depth_latent_velocity, 'velocity', noised_student_pred_depth_latent, sds_timestep)
                                frozen_denoiser_pred_depth = frozen_denoiser_pred_depth_latent + 0.5

                                # TODO figure out if the sign should be reversed here
                            score_vector = (pred_noise.float() - noise.float())

                            sds_loss = 0.5 * F.mse_loss(student_pred_depth_latent.float(), (student_pred_depth_latent.float() - score_vector.float()).detach().float(), reduction="mean")

                        # our custom losses! MSE depth loss far from edges, and *high-frequency* MSE depth loss near edges
                        else:
                            depth_loss = F.mse_loss(student_pred_depth * is_far_from_edges_mask, target_depth.float() * is_far_from_edges_mask, reduction="mean").to(weight_dtype) / (is_far_from_edges_mask.float().mean() + 1e-6)
                            edge_loss = F.mse_loss(high_frequency_initial_depth * is_near_edge_mask, high_frequency_student_pred_depth * is_near_edge_mask, reduction="mean").to(weight_dtype) / (is_near_edge_mask.float().mean() + 1e-6)
                            sds_loss = edge_loss
                    else:
                        raise ValueError(f"Unknown conditioning kind: {conditioning_kind}")



                    # set variables needed for logging+visualization
                    depth_mse = depth_loss
                    initial_depth_loss = None
                    initial_depth_mse = None
                    
                    with torch.no_grad():
                        final_pred_depth_aligned, _, _ = align_depth_least_square(
                            gt_arr=target_depth.detach().float().cpu().numpy(),
                            pred_arr=student_pred_depth.detach().float().cpu().numpy(),
                            valid_mask_arr=torch.ones_like(student_pred_depth).detach().bool().cpu().numpy(),
                            return_scale_shift=True,
                            max_resolution=None,
                        )
                        final_pred_depth_aligned = torch.from_numpy(final_pred_depth_aligned).to(device)
                        final_aligned_depth_mse = F.mse_loss(final_pred_depth_aligned, target_depth, reduction="mean")
                        
                        final_generation_aligned_depth_mse = final_aligned_depth_mse

                    if accelerator.is_main_process and args.log_depth_maps and step % 10 == 0:
                        with torch.no_grad():

                            os.makedirs("/tmp/viz", exist_ok=True)

                            Image.fromarray(((rgb * 255.0).int().squeeze(0,1).permute(1,2,0).cpu().numpy().astype(np.uint8))).save("/tmp/viz_rgb.png")

                            def colorize_internal(value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"):
                                colored = colorize_depth_maps(value.squeeze(0), vmin, vmax, cmap)
                                colored = (colored * 255).astype(np.uint8)
                                colored_hwc = chw2hwc(colored.squeeze(0))
                                return Image.fromarray(colored_hwc)
                            
                            
                            # sds_score = score_vector.abs()
                            # sds_score_colored = colorize_internal(sds_score.cpu().numpy(), sds_score.min().item(), sds_score.max().item(), cmap="coolwarm")
                            # sds_score_colored.save("/tmp/viz/sds_score.png")

                            # l1_error_colored = colorize_internal(maybe_blur(l1_error).cpu().numpy(), 0, 1, cmap="coolwarm")
                            # l1_error_colored.save("/tmp/viz/l1_error.png")
                            # print("l1_error.min(), l1_error.max()", l1_error.min().item(), l1_error.max().item())

                            # weighted_sds_score = (sds_score - maybe_blur(sds_score)).abs() * ((1-maybe_blur(l1_error))**2)
                            # weighted_sds_score_colored = colorize_internal(weighted_sds_score.cpu().numpy(), weighted_sds_score.min().item(), weighted_sds_score.max().item(), cmap="Greys")
                            # weighted_sds_score_colored.save("/tmp/viz/weighted_sds_score.png")

                            # base_depth_colored = colorize_internal(norm_base_depth.float().cpu().numpy(), norm_base_depth.min().item(), norm_base_depth.max().item(), cmap="coolwarm")
                            # base_depth_colored.save("/tmp/viz/base_depth.png")

                            initial_depth_colored = colorize_internal(frozen_pred_depth.float().cpu().numpy(), frozen_pred_depth.float().min().item(), frozen_pred_depth.float().max().item(), cmap="coolwarm")
                            initial_depth_colored.save("/tmp/viz/initial_depth.png")

                            target_depth_colored = colorize_internal(target_depth.float().cpu().numpy(), target_depth.float().min().item(), target_depth.float().max().item(), cmap="coolwarm")
                            target_depth_colored.save("/tmp/viz/target_depth.png")

                            blurred_target_depth_colored = colorize_internal(blurred_target_depth.float().cpu().numpy(), blurred_target_depth.float().min().item(), blurred_target_depth.float().max().item(), cmap="coolwarm")
                            blurred_target_depth_colored.save("/tmp/viz/blurred_target_depth.png")

                            diff = (student_pred_depth - target_depth).abs().float()
                            diff_colored = colorize_internal(diff.cpu().numpy(), diff.min().item(), diff.max().item(), cmap="coolwarm")
                            diff_colored.save("/tmp/viz/diff.png")

                            diff_far_from_edges = (student_pred_depth - target_depth).abs().float() * is_far_from_edges_mask.float()
                            diff_far_from_edges_colored = colorize_internal(diff_far_from_edges.cpu().numpy(), diff_far_from_edges.min().item(), diff_far_from_edges.max().item(), cmap="coolwarm")
                            diff_far_from_edges_colored.save("/tmp/viz/diff_far_from_edges.png")

                            high_frequency_diff_near_edges = (high_frequency_initial_depth - high_frequency_student_pred_depth).abs().float() * is_near_edge_mask.float()
                            high_frequency_diff_near_edges_colored = colorize_internal(high_frequency_diff_near_edges.cpu().numpy(), high_frequency_diff_near_edges.min().item(), high_frequency_diff_near_edges.max().item(), cmap="coolwarm")
                            high_frequency_diff_near_edges_colored.save("/tmp/viz/high_frequency_diff_near_edges.png")

                            is_far_from_edges_mask_colored = colorize_internal(is_far_from_edges_mask.float().cpu().numpy(), 0.0, 1.0, cmap="coolwarm")
                            is_far_from_edges_mask_colored.save("/tmp/viz/is_far_from_edges_mask.png")

                            final_depth_colored = colorize_internal(student_pred_depth.float().cpu().numpy(), student_pred_depth.float().min().item(), student_pred_depth.float().max().item(), cmap="coolwarm")
                            final_depth_colored.save("/tmp/viz/final_depth.png")

                            # frozen_denoiser_pred_depth_colored = colorize_internal(frozen_denoiser_pred_depth.cpu().numpy(), frozen_denoiser_pred_depth.min().item(), frozen_denoiser_pred_depth.max().item(), cmap="coolwarm")
                            # frozen_denoiser_pred_depth_colored.save("/tmp/viz/frozen_denoiser_pred_depth.png")

                            # sds_input_depth = colorize_internal()

                            rgb_img = Image.fromarray(((rgb * 255.0).int().squeeze(0,1).permute(1,2,0).cpu().numpy().astype(np.uint8)))

                            # let's concatenate them vertically!
                            concatenated = np.concatenate([np.array(img) for img in [
                                initial_depth_colored,
                                target_depth_colored,
                                blurred_target_depth_colored,
                                diff_colored,
                                diff_far_from_edges_colored,
                                is_far_from_edges_mask_colored,
                                final_depth_colored,
                                rgb_img,
                            ]],axis=0)
                            Image.fromarray(concatenated).save("/tmp/viz/concatenated.png")

                            pass

                else:
                    raise NotImplementedError(f"Image resizing not implemented for denoiser={sharpdepth_kind}")

                # ------------------------------------------------------------------
                # Depth loss

                # ------------------------------------------------------------------
                # Optimization
                loss_weight_sum = args.sds_loss_weight + args.depth_weight 
                if args.use_normal_loss:
                    loss_weight_sum += args.normal_loss_weight
                    args.normal_loss_weight /= loss_weight_sum
                args.sds_loss_weight /= loss_weight_sum
                args.depth_weight /= loss_weight_sum

                loss = sds_loss * args.sds_loss_weight + depth_loss * args.depth_weight 
                if args.use_normal_loss:
                    loss += normal_loss * args.normal_loss_weight

                loss_accum += loss.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:

                    loss_accum /= accelerator.gradient_accumulation_steps
                    loss_accum = accelerator.reduce(loss_accum, reduction="mean")

                    last_loss_accum = loss_accum
                    loss_accum = torch.tensor(0.0, device=device, dtype=loss.dtype)

                    if args.use_ema:
                        ema_model.update()

                    params_to_clip = student_denoiser.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                # ------------------------------------------------------------------

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints) - args.checkpoints_total_limit + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if global_step % args.validation_steps == 0 or global_step == 1:

                            student_denoiser.eval()
                            saved_dir = os.path.join(
                                output_dir, "visualization", f"iter_{global_step}"
                            )
                            os.makedirs(saved_dir, exist_ok=True)

                            default_denoising_steps = {
                                SharpDepthKind.LOTUS: 1,
                                SharpDepthKind.PIXEL_PERFECT_DEPTH: 4,
                                SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET: 4,
                            }[sharpdepth_kind]

                            default_processing_resolution = {
                                SharpDepthKind.LOTUS: 768,
                                SharpDepthKind.PIXEL_PERFECT_DEPTH: 1024,
                                SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET: 1024,
                            }[sharpdepth_kind]

                            pipeline = SharpDepthPipeline.from_pretrained(
                                student_ckpt_dir,
                                unet=unwrap_model(student_denoiser),
                                frozen_unet=unwrap_model(frozen_denoiser),
                                vae=unwrap_model(vae),
                                scheduler=noise_scheduler,
                                default_processing_resolution=default_processing_resolution,
                                default_denoising_steps=default_denoising_steps,
                                sharpdepth_kind=sharpdepth_kind,
                                base_depth_estimator_fn=base_depth_estimator_fn,
                                blur_difference_map_scale_factor=args.blur_unidepth_output_ratio,
                                noise_aware_latent_noise_scale=args.noise_aware_latent_noise_scale,
                                use_conditioning_for_initial_ppd=args.use_conditioning_for_initial_ppd,
                                initialize_ppd_from_timestep=args.initialize_ppd_from_timestep,
                                align_depth_least_square=args.align_depth_least_square,
                            ).to(accelerator.device, dtype=weight_dtype)
                            avg_rmse = 0.0
                            avg_rmse_base = 0.0
                            avg_rmse_initial = 0.0
                            total_images = 0
                            with torch.no_grad():
                                images = []
                                for loader_idx, loader in enumerate(val_loaders):
                                    images.append([])
                                    for vis_idx, batch in enumerate(loader):
                                        vis_imgs = {}
                                        images[loader_idx].append(vis_imgs)
                                        if vis_idx > 10:
                                            continue
                                        rgb = Image.fromarray(
                                            batch["rgb_int"]
                                            .squeeze()
                                            .permute(1, 2, 0)
                                            .cpu()
                                            .numpy()
                                            .astype(np.uint8)
                                        )
                                        rgb_int_1chw = torch.from_numpy(np.array(rgb)).to(torch.int32).permute(2, 0, 1).unsqueeze(0)
                                        out = pipeline(
                                            rgb_int_1chw, base_depth_estimator_fn, processing_res=768, denoising_steps=1
                                        )

                                        depth_pred = torch.from_numpy(out.depth_np).to(
                                            accelerator.device
                                        )
                                        depth_base_np = torch.from_numpy(out.depth_base_np).to(
                                            accelerator.device
                                        )
                                        depth_initial_np = torch.from_numpy(out.depth_initial_np).to(
                                            accelerator.device
                                        )

                                        depth_raw_linear_np = batch["depth_raw_linear"].squeeze().cpu().numpy()
                                        valid_mask_raw_np = batch["valid_mask_raw"].squeeze().cpu().numpy()
                                        rmse_final = rmse(out.depth_np, depth_raw_linear_np, valid_mask_raw_np)
                                        rmse_base = rmse(out.depth_base_np, depth_raw_linear_np, valid_mask_raw_np)
                                        rmse_initial = rmse(out.depth_initial_np, depth_raw_linear_np, valid_mask_raw_np)
                                        avg_rmse += rmse_final
                                        avg_rmse_base += rmse_base
                                        avg_rmse_initial += rmse_initial
                                        total_images += 1

                                        gt = (
                                            batch["depth_raw_linear"]
                                            .squeeze()
                                            .to(accelerator.device)
                                        )
                                        valid_mask = (
                                            batch["valid_mask_raw"].squeeze().to(accelerator.device)
                                        )

                                        error = abs_relative_difference_full(
                                            depth_pred, gt, valid_mask
                                        )
                                        error_uni = abs_relative_difference_full(
                                            depth_base_np, gt, valid_mask
                                        )

                                        error_col = colorize(
                                            error.cpu().numpy(), 0, 0.12, cmap="coolwarm"
                                        )
                                        error_uni_col = colorize(
                                            error_uni.cpu().numpy(), 0, 0.12, cmap="coolwarm"
                                        )

                                        if args.report_to == "wandb":
                                            wandb_tracker = accelerator.get_tracker("wandb")

                                        Image.fromarray(error_uni_col).save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_base_error_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        vis_imgs["vis_base_error"] = Image.fromarray(error_uni_col)

                                        Image.fromarray(error_col).save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_final_error_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        vis_imgs["vis_final_error"] = Image.fromarray(error_col)

                                        out.depth_colored.save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_final_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        vis_imgs["vis_final"] = out.depth_colored
                                        out.depth_base_colored.save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_base_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        vis_imgs["vis_base"] = out.depth_base_colored

                                        out.depth_initial_colored.save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_initial_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        vis_imgs["vis_initial"] = out.depth_initial_colored

                                        out.pred_mask.save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_diff_mask_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        vis_imgs["vis_diff_mask"] = out.pred_mask
                                        rgb.save(
                                            os.path.join(
                                                saved_dir, f"vis_rgb_{loader_idx}_{vis_idx}.jpg"
                                            )
                                        )
                                        vis_imgs["vis_rgb"] = rgb
                                    
                                # put them in a grid and log to wandb and filesystem
                                keys = set(images[0][0].keys())
                                wandb_log_obj = {}
                                for k in keys:
                                    img_list = []
                                    for loader_imgs in images:
                                        for vis_img_dict in loader_imgs:
                                            if k in vis_img_dict:
                                                img_list.append(vis_img_dict[k])
                                    
                                    if not img_list:
                                        continue
                                    
                                    n = len(img_list)
                                    cols = int(np.ceil(np.sqrt(n)))
                                    rows = int(np.ceil(n / cols))
                                    
                                    img_w, img_h = img_list[0].size
                                    
                                    grid = Image.new("RGB", size=(cols * img_w, rows * img_h))
                                    for idx, img in enumerate(img_list):
                                        row = idx // cols
                                        col = idx % cols
                                        grid.paste(img, (col * img_w, row * img_h))
                                    
                                    grid.save(os.path.join(saved_dir, f"grid_{k}.jpg"))
                                    wandb_log_obj[f"grid_{k}"] = wandb.Image(grid)
                                if args.report_to == "wandb":
                                    wandb_tracker.log(wandb_log_obj)
                            
                            avg_rmse /= total_images
                            avg_rmse_base /= total_images
                            avg_rmse_initial /= total_images
                            logs = {
                                "val_rmse": round(avg_rmse.item(), 4),
                                "val_rmse_base": round(avg_rmse_base.item(), 4),
                                "val_rmse_initial": round(avg_rmse_initial.item(), 4),
                            }
                            accelerator.log(logs, step=global_step)

                            del pipeline
                            torch.cuda.empty_cache()
                            student_denoiser.train()
                
                # we have a value for each of these losses, summed over 1 example (since our microbatch size is 1).
                # hence the (loss, 1) tuples.
                losses_and_counts = {
                    conditioning_kind:{
                        "total": (loss.detach(), torch.tensor(1,device=loss.device)),
                        "sds": (sds_loss.detach(), torch.tensor(1,device=sds_loss.device)),
                        "depth": (depth_loss.detach(), torch.tensor(1,device=depth_loss.device)),
                        "normal": (normal_loss.detach() if normal_loss is not None else torch.tensor(0,device=device), torch.tensor(1,device=normal_loss.device) if normal_loss is not None else torch.tensor(0,device=device)),
                        "depth_mse": (depth_mse.detach(), torch.tensor(1,device=depth_mse.device)),
                        "depth_aligned_mse": (final_aligned_depth_mse.detach(), torch.tensor(1,device=final_aligned_depth_mse.device)),
                        "initial_depth": (initial_depth_loss.detach() if initial_depth_loss is not None else torch.tensor(0,device=device), torch.tensor(1,device=initial_depth_loss.device) if initial_depth_loss is not None else torch.tensor(0,device=device)),
                        "initial_depth_mse": (initial_depth_mse.detach() if initial_depth_mse is not None else torch.tensor(0,device=device), torch.tensor(1,device=initial_depth_mse.device) if initial_depth_mse is not None else torch.tensor(0,device=device)),
                        "final_generation_depth_aligned_mse": (final_generation_aligned_depth_mse.detach(), torch.tensor(1,device=final_generation_aligned_depth_mse.device)),
                    },
                    "all": {
                        "total": (loss.detach(), torch.tensor(1,device=loss.device)),
                        "sds": (sds_loss.detach(), torch.tensor(1,device=sds_loss.device)),
                        "depth": (depth_loss.detach(), torch.tensor(1,device=depth_loss.device)),
                        "normal": (normal_loss.detach() if normal_loss is not None else torch.tensor(0,device=device), torch.tensor(1,device=normal_loss.device) if normal_loss is not None else torch.tensor(0,device=device)),
                        "depth_mse": (depth_mse.detach(), torch.tensor(1,device=depth_mse.device)),
                        "depth_aligned_mse": (final_aligned_depth_mse.detach(), torch.tensor(1,device=final_aligned_depth_mse.device)),
                        "initial_depth": (initial_depth_loss.detach() if initial_depth_loss is not None else torch.tensor(0,device=device), torch.tensor(1,device=initial_depth_loss.device) if initial_depth_loss is not None else torch.tensor(0,device=device)),
                        "initial_depth_mse": (initial_depth_mse.detach() if initial_depth_mse is not None else torch.tensor(0,device=device), torch.tensor(1,device=initial_depth_mse.device) if initial_depth_mse is not None else torch.tensor(0,device=device)),
                        "final_generation_depth_aligned_mse": (final_generation_aligned_depth_mse.detach(), torch.tensor(1,device=final_generation_aligned_depth_mse.device)),
                    },
                    **{empty_conditioning_kind:{loss_key:(torch.tensor(0,device=device,dtype=loss.dtype),torch.tensor(0,device=device,dtype=loss.dtype)) for loss_key in loss_keys} for empty_conditioning_kind in conditioning_kinds if empty_conditioning_kind not in [conditioning_kind, "all"]}
                }

                # average the losses and counts across devices!
                if accelerator.num_processes > 1:

                    accumulated_losses_and_counts = {}
                    for k in sorted(losses_and_counts.keys()):
                        accumulated_losses_and_counts[k] = {}
                        for loss_key in sorted(losses_and_counts[k].keys()):
                            raw_item = losses_and_counts[k][loss_key]
                            summed_item = accelerator.reduce((raw_item[0].to(torch.float32), raw_item[1].to(torch.int32)), reduction="sum")
                            accumulated_losses_and_counts[k][loss_key] = (summed_item[0] / summed_item[1] if summed_item[1] > 0 else torch.tensor(0.0, device=summed_item[0].device), summed_item[1])

                    losses_and_counts = accumulated_losses_and_counts
                
                update_loss_exponential_moving_averages(losses_and_counts)

                logs = {
                    "loss": round(last_loss_accum.cpu().item(), 6),
                    **{f"loss_{conditioning_kind}_{loss_key}": round(loss_exponential_moving_averages[conditioning_kind][loss_key].item(), 6) for conditioning_kind in conditioning_kinds for loss_key in loss_keys},
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    break
