# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Code adapted from:
# https://github.com/EnVision-Research/Lotus/blob/main/pipeline.py

import logging
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Union, Callable

from ppd_sharpdepth.ppd.utils.transform import resize_keep_aspect
from ppd_sharpdepth.sharpdepth.util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from ppd_sharpdepth.sharpdepth.util.normalizer import ScaleShiftNormalizer
from ppd_sharpdepth.sharpdepth.util.alignment import align_depth_least_square
from ppd_sharpdepth.ppd.models.ppd import PixelPerfectDepth

from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor

from ppd_sharpdepth.sharpdepth_kinds import SharpDepthKind

from ppd_sharpdepth.preprocessors import PreProcessor, MarigoldPreProcessor, PixelPerfectDepthPreProcessor

import torchvision.transforms.functional as TV_F

class SharpDepthOutput(BaseOutput):
    """
    Output class for Marigold Monocular Depth Estimation pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [H, W, 3] and values in [0, 255].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_base_np: np.ndarray
    depth_initial_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    depth_base_colored: Union[None, np.ndarray]
    depth_initial_colored: Union[None, np.ndarray]
    pred_mask: Union[None, Image.Image]

class SharpDepthPipeline(DiffusionPipeline):
    """
    Pipeline for Marigold Monocular Depth Estimation: https://marigoldcomputervision.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the prediction latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: Union[UNet2DConditionModel, PixelPerfectDepth],
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        frozen_unet: Optional[Union[UNet2DConditionModel, PixelPerfectDepth]] = None,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        sharpdepth_kind: Optional[SharpDepthKind] = None,
        base_depth_estimator_fn=None,
        blur_difference_map_scale_factor: int = 1,
        noise_aware_latent_noise_scale: float = 1.0,
        use_conditioning_for_initial_ppd: bool = False,
        initialize_ppd_from_timestep: Optional[int] = None,
        align_depth_least_square: Optional[Callable] = None,
    ):
        super().__init__()

        if not frozen_unet:

            if sharpdepth_kind in [SharpDepthKind.PIXEL_PERFECT_DEPTH, SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET]:
                raise ValueError("No frozen PPD denoiser provided. Bad news!")
            elif sharpdepth_kind == SharpDepthKind.LOTUS:
                print("\n"*20+"WARN: No frozen unet provided, using the student unet")
                frozen_unet = frozen_unet or unet
            else:
                raise ValueError(f"Unknown sharpdepth_kind: {sharpdepth_kind}")

        self.register_modules(
            unet=unet,
            frozen_unet=frozen_unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        assert default_processing_resolution is not None, "must pass default_processing_resolution to SharpDepthPipeline"
        assert default_denoising_steps is not None, "must pass default_denoising_steps to SharpDepthPipeline"
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        self.depth_normalizer = ScaleShiftNormalizer()
        self.image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)

        assert sharpdepth_kind is not None, "must pass sharpdepth_kind to SharpDepthPipeline"
        self.sharpdepth_kind = sharpdepth_kind

        assert base_depth_estimator_fn is not None, "must pass base_depth_estimator_fn to SharpDepthPipeline"
        self.base_depth_estimator_fn = base_depth_estimator_fn

        self.blur_difference_map_scale_factor = blur_difference_map_scale_factor

        self.noise_aware_latent_noise_scale = noise_aware_latent_noise_scale

        self.use_conditioning_for_initial_ppd = use_conditioning_for_initial_ppd

        self.initialize_ppd_from_timestep = initialize_ppd_from_timestep
        self.align_depth_least_square = align_depth_least_square

    @torch.no_grad()
    def __call__(
        self,
        rgb_int_1chw: torch.Tensor,
        intrinsics=None,
        denoising_steps: Optional[int] = None,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "inferno_r",
        show_progress_bar: bool = True,
    ) -> SharpDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, *optional*, defaults to `1`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize the prediction to match the input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or
                `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False,
                near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [H, W, 3] and values in [0, 255], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """

        assert isinstance(rgb_int_1chw, torch.Tensor), "rgb_int_1chw must be a torch.Tensor"
        assert rgb_int_1chw.dtype == torch.int32, "rgb_int_1chw must be of dtype torch.int32"

        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0

        self.encode_empty_text()
        input_size = rgb_int_1chw.shape
        assert (
            4 == rgb_int_1chw.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        if self.sharpdepth_kind == SharpDepthKind.LOTUS:
            depth_base = depth_base_11hw = self.base_depth_estimator_fn(rgb_int_1chw, MarigoldPreProcessor, internal=True)

            rgb_float_1chw_resized, padding, original_resolution = MarigoldPreProcessor.run(rgb_int_1chw, self.device, self.dtype)

            rgb_norm =  rgb_float_1chw_resized * 2.0 - 1.0  #  [0, 1] -> [-1, 1]
            rgb_norm = rgb_norm.to(self.dtype).to(self.device)

            normalize_obj = self.depth_normalizer(depth_base)
            norm_disp_base = normalize_obj['norm_depth'].to(dtype=self.vae.dtype)

            base_latent = self.encode_rgb(norm_disp_base.to(self.vae.dtype).repeat(1,3,1,1))
            rgb_latent = self.encode_rgb(rgb_norm.to(self.vae.dtype))
            lotus_timesteps = torch.ones((rgb_latent.shape[0],), device=self.device) * (self.scheduler.config.num_train_timesteps - 1)
            lotus_timesteps = lotus_timesteps.long()

            lotus_input = torch.cat([rgb_latent.detach(), torch.randn_like(rgb_latent)], dim=1)  # this order is important
            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).to(self.device, self.dtype)

            lotus_pred  = self.unet(lotus_input, lotus_timesteps.to(self.vae.dtype), self.empty_text_embed, class_labels=task_emb).sample

            # decode pred_latent to depth
            latent = lotus_pred / self.vae.config.scaling_factor
            z = self.vae.post_quant_conv(latent.to(self.vae.dtype))
            lotus_depth = self.vae.decoder(z).mean(dim=1, keepdim=True)
            lotus_depth_initial = lotus_depth
            
            # compute difference mask
            l1_error            = torch.abs(lotus_depth -  norm_disp_base)
            l1_error            = l1_error/l1_error.max()
            l1_error            = l1_error.clip(0, 1)
        
            latent_mask = torch.nn.functional.interpolate(l1_error, scale_factor=1/8)

            noise                   = torch.randn_like(base_latent).to(self.vae.device)
            noisy_lotus_latent      = self.scheduler.add_noise(lotus_pred, noise, lotus_timesteps) # noisy_lotus_latent ~= noise
            noisy_latent            = noisy_lotus_latent * latent_mask + base_latent * (1 - latent_mask) # noisy_latent ~= noise
            student_input           = torch.cat([rgb_latent, noisy_latent], dim=1)

            pred_latent     = self.unet(student_input, lotus_timesteps.to(self.vae.dtype), encoder_hidden_states=self.empty_text_embed, class_labels=task_emb).sample

            # decode pred_latent to depth
            latent = pred_latent / self.vae.config.scaling_factor
            z = self.vae.post_quant_conv(latent.to(self.vae.dtype))
            lotus_depth = self.vae.decoder(z).mean(dim=1, keepdim=True)

            student_pred_depth = lotus_depth
            depth_base_11hw = depth_base
            initial_pred = lotus_depth_initial
            l1_error = l1_error
        elif self.sharpdepth_kind == SharpDepthKind.PIXEL_PERFECT_DEPTH: 
            depth_base = depth_base_11hw = self.base_depth_estimator_fn(rgb_int_1chw, PixelPerfectDepthPreProcessor, internal=True)
            rgb_float_1chw_resized, padding, original_resolution = PixelPerfectDepthPreProcessor.run(rgb_int_1chw, self.device, self.dtype)

            normalize_obj = self.depth_normalizer(depth_base_11hw)
            norm_base_depth = normalize_obj["norm_depth"].to(dtype=self.unet.dtype)
            norm_base_depth = norm_base_depth * 0.5 + 0.5

            # initial PPD
            cond = rgb_float_1chw_resized - 0.5
            noise = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(self.device)
            with torch.autocast(self.device.type,dtype=self.unet.dtype):
                semantics = self.unet.semantics_prompt(rgb_float_1chw_resized)
                latent = noise
                cond_noise = (norm_base_depth - 0.5) if self.use_conditioning_for_initial_ppd else torch.randn_like(latent)
                for timestep in self.unet.sampling_timesteps:
                    input = torch.cat([latent, cond, cond_noise], dim=1)
                    pred = self.unet.dit(x=input, semantics=semantics, timestep=timestep)
                    latent = self.unet.sampler.step(pred=pred, x_t=latent, t=timestep)
                initial_pred = latent + 0.5
        
            # compute difference mask
            l1_error = torch.abs(initial_pred - norm_base_depth)
            l1_error = l1_error / l1_error.max()
            l1_error = l1_error.clip(0, 1)
            l1_mask = l1_error

            scaled_l1_mask = l1_mask * self.noise_aware_latent_noise_scale
            noisy_depth_cond = (norm_base_depth - 0.5) * (1 - scaled_l1_mask) + (torch.randn_like(norm_base_depth) * scaled_l1_mask)

            # second PPD

            noise = torch.randn_like(noise)
            if self.initialize_ppd_from_timestep is not None:
                timesteps = torch.tensor([timestep for timestep in self.unet.sampling_timesteps if timestep <= self.initialize_ppd_from_timestep],device=self.device,dtype=self.unet.dtype)
                latent = self.unet.schedule.forward(norm_base_depth - 0.5, noise, torch.tensor(self.initialize_ppd_from_timestep,device=self.device,dtype=self.unet.dtype))
            else:
                timesteps = torch.tensor(self.unet.sampling_timesteps,device=self.device,dtype=self.unet.dtype)
                latent = noise

            with torch.autocast(self.device.type,dtype=self.unet.dtype):
                for timestep in timesteps:
                    student_input = torch.cat([latent, cond, noisy_depth_cond], dim=1)
                    pred = self.unet.dit(x=student_input, semantics=semantics, timestep=timestep)
                    latent = self.unet.sampler.step(pred=pred, x_t=latent, t=timestep)
                student_pred_depth = latent + 0.5
        
        elif self.sharpdepth_kind == SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET:
            depth_base = depth_base_11hw = self.base_depth_estimator_fn(rgb_int_1chw, PixelPerfectDepthPreProcessor, internal=True)
            rgb_float_1chw_resized, padding, original_resolution = PixelPerfectDepthPreProcessor.run(rgb_int_1chw, self.device, self.dtype)

            normalize_obj = self.depth_normalizer(depth_base_11hw)
            norm_base_depth = normalize_obj["norm_depth"].to(dtype=self.unet.dtype)
            norm_base_depth = norm_base_depth * 0.5 + 0.5

            # initial PPD
            cond = rgb_float_1chw_resized - 0.5
            noise = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(self.device)
            with torch.autocast(self.device.type,dtype=self.unet.dtype):
                semantics = self.frozen_unet.semantics_prompt(rgb_float_1chw_resized)
                latent = noise
                for timestep in self.frozen_unet.sampling_timesteps:
                    input = torch.cat([latent, cond], dim=1)
                    pred = self.frozen_unet.dit(x=input, semantics=semantics, timestep=timestep)
                    latent = self.frozen_unet.sampler.step(pred=pred, x_t=latent, t=timestep)
                initial_pred = latent + 0.5
        
            # compute difference mask
            l1_error = torch.abs(initial_pred - norm_base_depth)
            l1_error = l1_error / l1_error.max()
            l1_error = l1_error.clip(0, 1)
            l1_mask = l1_error

            scaled_l1_mask = l1_mask * self.noise_aware_latent_noise_scale
            # noisy_depth_cond = (norm_base_depth - 0.5) * (1 - scaled_l1_mask) + (torch.randn_like(norm_base_depth) * scaled_l1_mask)

            # PPD with controlnet

            # let's blur the base prediction!
            def blur(x_11hw, scale_factor):
                return TV_F.gaussian_blur(x_11hw, kernel_size=2*(scale_factor//2)+1, sigma=scale_factor/2)
            def maybe_blur(x_11hw):
                assert self.blur_difference_map_scale_factor > 1, f"blur_difference_map_scale_factor must be greater than 1, got {self.blur_difference_map_scale_factor}"
                return blur(x_11hw, self.blur_difference_map_scale_factor)
            blurred_base_depth = maybe_blur(norm_base_depth)

            noise = torch.randn_like(noise)
            cond_inputs = [
                torch.cat([
                    blurred_base_depth - 0.5,
                    cond,
                ], dim=1),
                torch.cat([
                    initial_pred - 0.5,
                    cond,
                ], dim=1),
            ]

            timesteps = torch.tensor(self.unet.sampling_timesteps,device=self.device,dtype=self.unet.dtype)
            latent = noise

            with torch.autocast(self.device.type,dtype=self.unet.dtype):
                for timestep in timesteps:
                    student_input = torch.cat([latent, cond], dim=1)
                    pred = self.unet.dit(x=student_input, conds=cond_inputs, semantics=semantics, timestep=timestep)
                    latent = self.unet.sampler.step(pred=pred, x_t=latent, t=timestep)
                student_pred_depth = latent + 0.5
            
        else:
            raise NotImplementedError(f"SharpDepthKind {self.sharpdepth_kind} not implemented yet")

        
        final_pred = self.image_processor.unpad_image(student_pred_depth, padding)  # [N*E,1,PH,PW]
        base_pred = self.image_processor.unpad_image(depth_base_11hw, padding)  # [N*E,1,PH,PW]
        initial_pred = self.image_processor.unpad_image(initial_pred, padding)  # [N*E,1,PH,PW]
        l1_error = self.image_processor.unpad_image(l1_error, padding)  # [N*E,1,PH,PW]
        
        final_pred = self.image_processor.resize_antialias(final_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
        base_pred = self.image_processor.resize_antialias(base_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
        initial_pred = self.image_processor.resize_antialias(initial_pred, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]
        l1_error = self.image_processor.resize_antialias(l1_error, original_resolution, mode="bilinear", is_aa=False)  # [N,1,H,W]

        # Convert to numpy
        final_pred = final_pred.squeeze().float().cpu().numpy()
        base_pred = base_pred.squeeze().float().cpu().numpy()
        initial_pred = initial_pred.squeeze().float().cpu().numpy()
        pred_mask = l1_error.squeeze().float().cpu().numpy()

        valid_mask = (1 - pred_mask) > 0.5
        
        if self.align_depth_least_square:

            valid_mask = np.zeros_like(base_pred, dtype=bool)
            valid_mask[8:-8, 8:-8] = True

            final_pred, scale, shift = align_depth_least_square(
                    gt_arr=base_pred,
                    pred_arr=final_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=None,
            )
        else:
            if self.sharpdepth_kind == SharpDepthKind.LOTUS:
                final_pred = (final_pred + 1) / 2 * (normalize_obj['max'].item() - normalize_obj['min'].item()) + normalize_obj['min'].item()
            elif self.sharpdepth_kind in [SharpDepthKind.PIXEL_PERFECT_DEPTH, SharpDepthKind.PIXEL_PERFECT_DEPTH_CONTROLNET]:
                final_pred = final_pred * (normalize_obj['max'].item() - normalize_obj['min'].item()) + normalize_obj['min'].item()
            else:
                raise NotImplementedError(f"SharpDepthKind {self.sharpdepth_kind} not implemented yet")

        initial_pred, scale, shift = align_depth_least_square(
                                                        gt_arr=base_pred,
                                                        pred_arr=initial_pred,
                                                        valid_mask_arr=valid_mask,
                                                        return_scale_shift=True,
                                                        max_resolution=None,
                                                )

        
        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(final_pred, 0, base_pred.max(), cmap=color_map).squeeze()
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)

            depth_colored = colorize_depth_maps(base_pred, 0, base_pred.max(), cmap=color_map).squeeze()
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_base_colored_img = Image.fromarray(depth_colored_hwc)

            depth_colored = colorize_depth_maps(initial_pred, 0, initial_pred.max(), cmap=color_map).squeeze()
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_initial_colored_img = Image.fromarray(depth_colored_hwc)

            depth_colored = colorize_depth_maps(pred_mask, 0, 1, cmap="coolwarm").squeeze() 
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            pred_mask_img = Image.fromarray(depth_colored_hwc)

        else:
            depth_colored_img = None
            depth_base_colored_img = None
            depth_initial_colored_img = None
            pred_mask_img = None

        return SharpDepthOutput(
            depth_np=final_pred,
            depth_base_np=base_pred,
            depth_initial_np=initial_pred,
            depth_colored=depth_colored_img,
            depth_base_colored=depth_base_colored_img,
            depth_initial_colored=depth_initial_colored_img,
            pred_mask=pred_mask_img,
        )

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

