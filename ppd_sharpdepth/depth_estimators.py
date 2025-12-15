from enum import Enum
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, Callable
from PIL import Image
import sys
import os
from mmengine.config import Config
import gc

from ppd_sharpdepth.ppd.models.dit import ControlNetDiT
from diffusers import UNet2DConditionModel
from ppd_sharpdepth.sharpdepth.util.alignment import align_depth_least_square
from unidepth.models import UniDepthV1
from .sharpdepth.data.datasets_and_samplers import get_dataset
from .sharpdepth.data.datasets_and_samplers.base_depth_dataset import (
    BaseDepthDataset,
    DatasetMode,
    DepthFileNameMode,
    get_pred_name,
)

from .depth_anything.dpt import DepthAnything
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor

from huggingface_hub import hf_hub_download
from ppd_sharpdepth.ppd.models.ppd import PixelPerfectDepth

from .sharpdepth.pipeline.pipeline import SharpDepthPipeline
from .sharpdepth_kinds import SharpDepthKind

# Debugging mmengine related issues
# Initial thought: For PatchRefiner, ensure registry is fresh. Otherwise, this may lead to a KeyError when importing PatchRefiner.
# Result: Upon further debugging, this does not solve the problem.

#print(sys.modules.keys())
#for k in list(sys.modules.keys()):
#    if k.startswith("estimator."):
#        del sys.modules[k]

#from .patchrefiner.estimator.registry import MODELS
#MODELS.clear()
#from .patchrefiner.estimator.registry import DATASETS
#DATASETS.clear()

# Attempt to get PatchRefiner model from mmengine. This does not work since it doesn't exist.
#PatchRefiner = sys.modules["estimator.models.patchrefiner"]

# Moved patchrefiner imports to inside the function to prevent cyclic imports.
#from .patchrefiner.estimator.models.patchrefiner import PatchRefiner
#from .patchrefiner.checkpoints import download as download_checkpoints

import torch.nn.functional as F

from .preprocessors import PreProcessor, MarigoldPreProcessor, PixelPerfectDepthPreProcessor

ModelArchitecture = Enum(
    "ModelArchitecture",
    map(
        lambda model_name: (model_name, model_name),
        [
            "sharpdepth_lotus_unidepth",
            "sharpdepth_lotus_zoedepth",
            "sharpdepth_ppd_unidepth",
            "depthanythingsmall",
            "depthanythinglarge",
            "pixelperfectdepth_unidepth",
            "pixelperfectdepth_zoedepth",
            "unidepth",
            "patchrefiner",
            "zoedepth",
            "sharpdepth_ppd_controlnet_zoedepth",
            "sharpdepth_ppd_controlnet_unidepth",
            "sharpdepth_ppd_timestep_500_unidepth",
            "sharpdepth_ppd_timestep_500_zoedepth",
        ]
    )
)


def get_depth_estimator_fn(
        model_architecture: ModelArchitecture, 
        device: torch.device, 
        float_dtype: torch.dtype, 
        checkpoint_filepath: Path, 
) -> Callable[[torch.Tensor, Type[PreProcessor]], torch.Tensor]:
    match model_architecture:
        case (
            ModelArchitecture.sharpdepth_lotus_unidepth 
            | ModelArchitecture.sharpdepth_lotus_zoedepth
        ):
            match model_architecture:
                case ModelArchitecture.sharpdepth_lotus_unidepth:
                    BASE_MODEL_CHECKPOINT_DIR = "lpiccinelli/unidepth-v1-vitl14"
                    base_depth_estimator_fn = get_depth_estimator_fn(
                        ModelArchitecture.unidepth, 
                        device, 
                        float_dtype, 
                        BASE_MODEL_CHECKPOINT_DIR
                    )
                case ModelArchitecture.sharpdepth_lotus_zoedepth:
                    BASE_MODEL_CHECKPOINT_DIR = "LiheYoung/depth_anything_vits14"
                    base_depth_estimator_fn = get_depth_estimator_fn(
                        ModelArchitecture.zoedepth, 
                        device, 
                        float_dtype, 
                        "isl-org/ZoeDepth"
                    )

            pipeline = SharpDepthPipeline.from_pretrained(
                checkpoint_filepath, 
                sharpdepth_kind=SharpDepthKind.LOTUS, 
                base_depth_estimator_fn=base_depth_estimator_fn,
                default_processing_resolution=768, 
                default_denoising_steps=1,
                align_depth_least_square=True,
             )
            assert pipeline.default_processing_resolution == 768, f"default_processing_resolution = {pipeline.default_processing_resolution}, expected 768"
            assert pipeline.default_denoising_steps == 1, f"default_denoising_steps = {pipeline.default_denoising_steps}, expected 1"

            pipeline = pipeline.to(device, dtype=float_dtype)

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor], internal=False):
                out = pipeline(rgb_int_1chw)

                h, w = out.depth_np.shape
                ret_11hw = torch.from_numpy(out.depth_np.reshape(1, 1, h, w))
                
                return ret_11hw

        case (
            ModelArchitecture.sharpdepth_ppd_unidepth 
            | ModelArchitecture.sharpdepth_ppd_timestep_500_unidepth 
            | ModelArchitecture.sharpdepth_ppd_timestep_500_zoedepth
        ):

            initialize_ppd_from_timestep = 500 if model_architecture in [
                ModelArchitecture.sharpdepth_ppd_timestep_500_unidepth,
                ModelArchitecture.sharpdepth_ppd_timestep_500_zoedepth,
            ] else None

            default_denoising_steps = 2 if model_architecture in [
                ModelArchitecture.sharpdepth_ppd_timestep_500_unidepth,
                ModelArchitecture.sharpdepth_ppd_timestep_500_zoedepth,
            ] else 4

            #with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 

            if model_architecture in [ModelArchitecture.sharpdepth_ppd_unidepth, ModelArchitecture.sharpdepth_ppd_timestep_500_unidepth]:
                base_depth_estimator_fn = get_depth_estimator_fn(
                    ModelArchitecture.unidepth, 
                    device, 
                    float_dtype, 
                    "lpiccinelli/unidepth-v1-vitl14"
                )
            elif model_architecture in [ModelArchitecture.sharpdepth_ppd_timestep_500_zoedepth]:
                base_depth_estimator_fn = get_depth_estimator_fn(
                    ModelArchitecture.zoedepth, 
                    device, 
                    float_dtype, 
                    "isl-org/ZoeDepth"
                )
            else:
                raise ValueError(f"Unknown model architecture: {model_architecture}")


            frozen_unet = PixelPerfectDepth.from_pretrained("andrew-healey/sharpdepth", subfolder="ppd", revision="main")
            frozen_unet = frozen_unet.to(device, dtype=float_dtype).eval()
            frozen_unet.requires_grad_(False)

            student_unet = PixelPerfectDepth.from_pretrained(checkpoint_filepath, subfolder="ppd_student")
            student_unet = student_unet.to(device, dtype=float_dtype).eval()
            student_unet.requires_grad_(False)

            pipeline = SharpDepthPipeline.from_pretrained(
                "andrew-healey/sharpdepth", 
                sharpdepth_kind=SharpDepthKind.PIXEL_PERFECT_DEPTH, 
                base_depth_estimator_fn=base_depth_estimator_fn,
                default_processing_resolution=768, 
                default_denoising_steps=default_denoising_steps,
                frozen_unet=frozen_unet,
                unet=student_unet,
                blur_difference_map_scale_factor=32, 
                noise_aware_latent_noise_scale=0,
                use_conditioning_for_initial_ppd=True,
                initialize_ppd_from_timestep=initialize_ppd_from_timestep,
            )
            assert pipeline.default_processing_resolution == 768, f"default_processing_resolution = {pipeline.default_processing_resolution}, expected 768"
            # assert pipeline.default_denoising_steps == default_denoising_steps, f"default_denoising_steps = {pipeline.default_denoising_steps}, expected {default_denoising_steps}"

            pipeline = pipeline.to(device, dtype=float_dtype)

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor], internal=False):
                #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = pipeline(rgb_int_1chw)

                h, w = out.depth_np.shape
                ret_11hw = torch.from_numpy(out.depth_np.reshape(1, 1, h, w))
                
                return ret_11hw

        case (
            ModelArchitecture.sharpdepth_ppd_controlnet_zoedepth
            | ModelArchitecture.sharpdepth_ppd_controlnet_unidepth
        ):

            if model_architecture == ModelArchitecture.sharpdepth_ppd_controlnet_unidepth:
                base_depth_estimator_fn = get_depth_estimator_fn(
                    ModelArchitecture.unidepth, 
                    device, 
                    float_dtype, 
                    "lpiccinelli/unidepth-v1-vitl14"
                )
            elif model_architecture == ModelArchitecture.sharpdepth_ppd_controlnet_zoedepth:
                base_depth_estimator_fn = get_depth_estimator_fn(
                    ModelArchitecture.zoedepth, 
                    device, 
                    float_dtype, 
                    "isl-org/ZoeDepth"
                )
            else:
                raise ValueError(f"Unknown model architecture: {model_architecture}")


            frozen_unet = PixelPerfectDepth.from_pretrained("andrew-healey/sharpdepth", subfolder="ppd", revision="main")
            frozen_unet = frozen_unet.to(device, dtype=float_dtype).eval()
            frozen_unet.requires_grad_(False)

            student_unet = PixelPerfectDepth.from_pretrained(checkpoint_filepath, subfolder="ppd_student_controlnet")
            student_unet = student_unet.to(device, dtype=float_dtype).eval()
            student_unet.requires_grad_(False)

            pipeline = SharpDepthPipeline.from_pretrained(
                "andrew-healey/sharpdepth", 
                sharpdepth_kind=SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET, 
                base_depth_estimator_fn=base_depth_estimator_fn,
                default_processing_resolution=768, 
                default_denoising_steps=4,
                frozen_unet=frozen_unet,
                unet=student_unet,
                blur_difference_map_scale_factor=64, 
                noise_aware_latent_noise_scale=0,
                use_conditioning_for_initial_ppd=False,
                initialize_ppd_from_timestep=None,
            )
            assert pipeline.default_processing_resolution == 768, f"default_processing_resolution = {pipeline.default_processing_resolution}, expected 768"
            assert pipeline.default_denoising_steps == 4, f"default_denoising_steps = {pipeline.default_denoising_steps}, expected 4"

            pipeline = pipeline.to(device, dtype=float_dtype)

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor], internal=False):
                #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = pipeline(rgb_int_1chw)

                h, w = out.depth_np.shape
                ret_11hw = torch.from_numpy(out.depth_np.reshape(1, 1, h, w))
                
                return ret_11hw


        case ModelArchitecture.unidepth:

            unidepth = UniDepthV1.from_pretrained(checkpoint_filepath)
            unidepth = unidepth.to(device, dtype=float_dtype)
            unidepth.requires_grad_(False)

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor], internal=False):
                
                rgb_float_1chw_resized, padding, original_resolution = preprocessor.run(rgb_int_1chw, device, float_dtype)

                base_pred = unidepth.infer(
                    (rgb_float_1chw_resized * 255).squeeze().int()
                )["depth"]

                if internal:
                    #ret_11hw = torch.from_numpy(base_pred)
                    return base_pred

                image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)
                base_pred = image_processor.unpad_image(base_pred, padding)  # [N*E,1,PH,PW]
                base_pred = image_processor.resize_antialias(base_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
                base_pred = base_pred.squeeze().float().cpu().numpy()

                ret_11hw = torch.from_numpy(base_pred)

                return ret_11hw

                # sanity check, for reference:

                # print(f"Input to unidepth. shape: {marigold_preprocessed_image.shape}, std: {marigold_preprocessed_image.std()}, mean: {marigold_preprocessed_image.mean()}, dtype: {marigold_preprocessed_image.dtype}")
                # print(f"Output from unidepth. shape: {ret_11hw.shape}, std: {ret_11hw.std()}, mean: {ret_11hw.mean()}, dtype: {ret_11hw.dtype}")
                # raise ValueError("Stop here")

                # Input to unidepth. shape: torch.Size([1, 3, 728, 768]), std: 0.32351335883140564, mean: 0.34626907110214233, dtype: torch.float32
                # Output from unidepth. shape: torch.Size([1, 1, 728, 768]), std: 0.360894113779068, mean: 1.302354335784912, dtype: torch.float32

        case (
            ModelArchitecture.depthanythingsmall | ModelArchitecture.depthanythinglarge
        ): 

            depth_anything = (
                DepthAnything.from_pretrained(checkpoint_filepath)
                .to(device)
                .eval()
            )
            depth_anything.requires_grad_(False)

            transform = Compose(
                [
                    Resize(
                        width=518,
                        height=518,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    PrepareForNet(),
                ]
            )

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor], internal=False): 

                rgb_float_1chw_resized, padding, original_resolution = preprocessor.run(rgb_int_1chw, device, float_dtype)

                image_1hwc = transform(
                    {
                        "image": rgb_float_1chw_resized
                        .permute(0, 2, 3, 1)
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy()
                    }
                )["image"][None]
                image_1hwc = torch.from_numpy(image_1hwc).to(device)

                disparity_raw_1hw = depth_anything(image_1hwc)
                depth_raw_1hw = disparity_raw_1hw.max() - disparity_raw_1hw

                depth_resized_11hw = F.interpolate(
                    depth_raw_1hw[None],
                    (
                        rgb_float_1chw_resized.shape[-2],
                        rgb_float_1chw_resized.shape[-1],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                
                if internal:
                    return depth_resized_11hw

                image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)
                base_pred = image_processor.unpad_image(depth_resized_11hw, padding)  # [N*E,1,PH,PW]
                base_pred = image_processor.resize_antialias(base_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
                base_pred = base_pred.squeeze().float().cpu().numpy()

                ret_11hw = torch.from_numpy(base_pred)

                return ret_11hw


                # sanity check (Good! Matches the stats in submodules/Depth-Anything/run.py):
                # filename:  submodules/SharpDepth/assets/in-the-wild_example/00.jpg
                # Input to depth_anything. shape: torch.Size([1, 3, 518, 546]), std: 1.4334324598312378, mean: -0.45399439334869385, dtype: torch.float32
                # Output from depth_anything. shape: torch.Size([1, 1, 728, 768]), std: 7.16904878616333, mean: 10.526022911071777, dtype: torch.float32

                # print(f"Input to depth_anything. shape: {image_1hwc.shape}, std: {image_1hwc.std()}, mean: {image_1hwc.mean()}, dtype: {image_1hwc.dtype}")
                # print(f"Resized output from depth_anything. shape: {disparity_raw_1hw.shape}, std: {disparity_raw_1hw.std()}, mean: {disparity_raw_1hw.mean()}, dtype: {disparity_raw_1hw.dtype}")
                # raise NotImplementedError("Depth Anything is not implemented yet")

        case (
            ModelArchitecture.pixelperfectdepth_unidepth
            | ModelArchitecture.pixelperfectdepth_zoedepth
        ): 

            #DEPTH_ANYTHING_SMALL_CHECKPOINT_DIR="LiheYoung/depth_anything_vits14"
            #depth_anything_small_fn = get_depth_estimator_fn(
            #    ModelArchitecture.depthanythingsmall,
            #    device,
            #    float_dtype,
            #    DEPTH_ANYTHING_SMALL_CHECKPOINT_DIR
            #)

            if model_architecture == ModelArchitecture.pixelperfectdepth_unidepth:
                base_depth_estimator_fn = get_depth_estimator_fn(
                    ModelArchitecture.unidepth,
                    device,
                    float_dtype,
                    "lpiccinelli/unidepth-v1-vitl14"
                )
            else:
                base_depth_estimator_fn = get_depth_estimator_fn(
                    ModelArchitecture.zoedepth,
                    device,
                    float_dtype,
                    "isl-org/ZoeDepth"
                )


            model = PixelPerfectDepth.from_pretrained(
                checkpoint_filepath, subfolder="ppd"
            )
            model = model.to(device).eval()
            model.requires_grad_(False)

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor], internal=False):
                preprocessor = PixelPerfectDepthPreProcessor
 
                rgb_float_1chw_resized, padding, original_resolution = preprocessor.run(rgb_int_1chw, device, float_dtype)
                rgb_int_1chw_resized = (rgb_float_1chw_resized * 255).to(torch.int32)

                #metric_depth_base = depth_anything_small_fn(rgb_int_1chw, preprocessor, internal=True)
                #metric_depth_base = unidepth_fn(rgb_int_1chw, MarigoldPreProcessor, internal=True)
                metric_depth_base = base_depth_estimator_fn(rgb_int_1chw, preprocessor, internal=True)

                H, W = rgb_float_1chw_resized.squeeze(0).shape[1:3]
                raw_image_hwc = (
                    rgb_int_1chw_resized.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                )
                raw_image_hwc_bgr = cv2.cvtColor(
                    raw_image_hwc, cv2.COLOR_RGB2BGR
                )  # infer_image() applies a BGR->RGB conversion, so we must first convert from RGB->BGR here
                depth_raw_11hw, _ = model.infer_image(raw_image_hwc_bgr)
                depth_11hw = F.interpolate(
                    depth_raw_11hw, size=(H, W), mode="bilinear", align_corners=False
                )

                # depth_11hw is in log space
                # so let's use least-squares to align as closely as possible with log-space metric_depth_base
                # and then convert it to metric space!
                metric_depth_base_log_space = torch.log(metric_depth_base + 1)
                depth_11hw_aligned, _, _ = align_depth_least_square(
                    gt_arr=metric_depth_base_log_space.detach().float().cpu().numpy(),
                    pred_arr=depth_11hw.detach().float().cpu().numpy(),
                    valid_mask_arr=torch.ones_like(depth_11hw).bool().cpu().numpy(),
                    return_scale_shift=True,
                    max_resolution=None,
                )
                depth_11hw_aligned = torch.from_numpy(depth_11hw_aligned).to(device)

                depth_11hw_aligned = torch.exp(depth_11hw_aligned) - 1

                if internal:
                    return depth_11hw_aligned

                image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)
                base_pred = image_processor.unpad_image(depth_11hw_aligned, padding)  # [N*E,1,PH,PW]
                base_pred = image_processor.resize_antialias(base_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
                base_pred = base_pred.squeeze().float()

                ret_11hw = base_pred

                return ret_11hw


                # sanity check (passes!)
                # Input to pixel perfect depth. shape: (804, 848, 3), std: 82.61926369403086, mean: 88.29934251697487, dtype: uint8
                # Output from pixel perfect depth. shape: (1, 1, 3, 804), std: 0.3026374280452728, mean: 0.4216078519821167, dtype: float32

                # print(f"Input to pixel perfect depth. shape: {raw_image_hwc.shape}, std: {raw_image_hwc.std()}, mean: {raw_image_hwc.mean()}, dtype: {raw_image_hwc.dtype}")
                # print(f"Resized output from pixel perfect depth. shape: {depth_11hw.shape}, std: {depth_11hw.std()}, mean: {depth_11hw.mean()}, dtype: {depth_11hw.dtype}")
                # raise NotImplementedError("Pixel Perfect Depth is not implemented yet")
        
        case ModelArchitecture.zoedepth:
            
            try:
                zoedepth_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
            except Exception as e:
                torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)
                zoedepth_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
            zoedepth_n = zoedepth_n.to(device).eval()

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor], internal=False):
                rgb_float_1chw_resized, padding, original_resolution = preprocessor.run(rgb_int_1chw, device, float_dtype)
                depth_11hw = zoedepth_n.infer(rgb_float_1chw_resized)

                if internal:
                    return depth_11hw

                image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)
                base_pred = image_processor.unpad_image(depth_11hw, padding)  # [N*E,1,PH,PW]
                base_pred = image_processor.resize_antialias(base_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
                base_pred = base_pred.squeeze().float()

                ret_11hw = base_pred
                return ret_11hw

        case ModelArchitecture.patchrefiner:
            from .patchrefiner.estimator.models.patchrefiner import PatchRefiner
            from .patchrefiner.checkpoints import download as download_checkpoints

            CONFIG = "./ppd_sharpdepth/patchrefiner/configs/patchrefiner_zoedepth/pr_u4k.py"
            COARSE_CHECKPOINT = "./ppd_sharpdepth/patchrefiner/checkpoints/work_dir/zoedepth/u4k/coarse_pretrain/checkpoint_24.pth"
            FINE_CHECKPOINT = "./ppd_sharpdepth/patchrefiner/checkpoints/work_dir/zoedepth/u4k/pr/checkpoint_36.pth"
            
            if not (os.path.exists(COARSE_CHECKPOINT) and os.path.exists(FINE_CHECKPOINT)):
                print("Checkpoint files are missing: Downloading...")
                download_checkpoints()

            # Load config file
            cfg = Config.fromfile(CONFIG)

            # Build the model
            patchrefiner = PatchRefiner(cfg.model.config)
            #patchrefiner = PatchRefiner.from_pretrained(cfg.model.config)
            print("Model instantiated!")

            # This step is done within PatchRefiner.from_pretrained()
            # Load coarse branch checkpoint first
            coarse_ckpt = torch.load(COARSE_CHECKPOINT, map_location='cpu', weights_only=False)
            #print(f"Coarse checkpoint patch_process_shape: {coarse_ckpt['model_state_dict'].keys()}")
            patchrefiner.coarse_branch.load_state_dict(coarse_ckpt['model_state_dict'], strict=True)
            print("Coarse branch loaded!")

            # Load fine branch checkpoint
            fine_ckpt = torch.load(FINE_CHECKPOINT, weights_only=False)
            #print(f"Fine checkpoint patch_process_shape: {fine_ckpt['model_state_dict']['tile_cfg']}")
            patchrefiner.load_state_dict(fine_ckpt['model_state_dict'], strict=False)
            print("Fine branch loaded!")

            # Delete loaded checkpoints to free memory
            del coarse_ckpt
            del fine_ckpt
            gc.collect()
            torch.cuda.empty_cache()

            #print(f"PATCH_SHAPE: {patchrefiner.patch_process_shape}")

            # Change to eval mode
            patchrefiner.eval()
            print("Switched to eval mode!")

            patchrefiner = patchrefiner.to(device, dtype=float_dtype)
            patchrefiner.requires_grad_(False)

            @torch.autocast(device_type=device.type, dtype=float_dtype)
            def depth_estimator_fn(rgb_int_1chw: torch.Tensor, preprocessor: Type[PreProcessor]):
                
                rgb_float_1chw_resized, padding, original_resolution = preprocessor.run(rgb_int_1chw, device, float_dtype)

                #rgb_int_1chw_resized = (rgb_float_1chw_resized * 255).int()
                # Expects a float tensor
                """
                tile_temp = {}
                coarse_temp_dict = {
                    'coarse_depth_roi': torch.zeros(1, 1, rgb_float_1chw_resized.shape[2], rgb_float_1chw_resized.shape[3], device=device),
                    'coarse_feats_roi': torch.zeros(1, rgb_float_1chw_resized.shape[1], rgb_float_1chw_resized.shape[2], rgb_float_1chw_resized.shape[3], device=device)
                }

                ret_11hw = patchrefiner.infer_forward(rgb_float_1chw_resized, None, tile_temp, coarse_temp_dict)
                """
                #print(f"RGB_SHAPE: {rgb_float_1chw_resized.shape}")

                # Upscale image to match default tile_cfg
                #rgb_float_1chw_resized = F.interpolate(rgb_float_1chw_resized, size=(1080, 1920), mode="bilinear", align_corners=False)
                _, _, H, W = rgb_float_1chw_resized.shape
                tile_cfg = {
                    'image_raw_shape': (H, W),
                    'patch_split_num': (2, 2),
                }
                # Debug
                #print(f"Tile config: {tile_cfg}")

                base_pred, _ = patchrefiner.forward(mode='infer', image_lr=rgb_float_1chw_resized, image_hr=rgb_float_1chw_resized, tile_cfg=tile_cfg, process_num=1)#, cai_mode="m2")

                image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)
                base_pred = image_processor.unpad_image(base_pred, padding)  # [N*E,1,PH,PW]
                base_pred = image_processor.resize_antialias(base_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
                base_pred = base_pred.squeeze().float()

                ret_11hw = base_pred

                return ret_11hw
            
        case _:
            raise ValueError(f"Invalid model architecture: {model_architecture}")

    return depth_estimator_fn
