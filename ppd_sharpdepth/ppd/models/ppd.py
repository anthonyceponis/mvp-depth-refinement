from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from ppd_sharpdepth.ppd.utils.timesteps import Timesteps
from ppd_sharpdepth.sharpdepth.util.image_util import colorize_depth_maps, chw2hwc
from ppd_sharpdepth.ppd.utils.schedule import LinearSchedule
from ppd_sharpdepth.ppd.utils.sampler import EulerSampler
from ppd_sharpdepth.ppd.utils.transform import image2tensor, resize_1024, resize_1024_crop, resize_keep_aspect

from ppd_sharpdepth.ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from ppd_sharpdepth.ppd.models.dit import ControlNetDiT, DiT

from huggingface_hub import PyTorchModelHubMixin
from diffusers import ConfigMixin, ModelMixin

from typing import List, Any, Dict, Union, Optional

class PixelPerfectDepth(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    config_name = "config.json"

    def __init__(
        self,
        semantics_pth:Optional[str]=None,
        sampling_steps:int=4,
        depth_anything_v2_encoder:str='vitl',
        depth_anything_v2_features:int=256,
        depth_anything_v2_out_channels:List[int]=[256, 512, 1024, 1024],
        dit_in_channels:int=4,
        num_control_nets:int=0,
    ):
        super(PixelPerfectDepth, self).__init__()

        self.semantics_encoder = DepthAnythingV2(
            encoder=depth_anything_v2_encoder,
            features=depth_anything_v2_features,
            out_channels=depth_anything_v2_out_channels
        )

        if semantics_pth is not None:
            self.semantics_encoder.load_state_dict(torch.load(semantics_pth, map_location='cpu'), strict=False)
        self.semantics_encoder = self.semantics_encoder.eval()
        if num_control_nets > 0:
            self.dit = ControlNetDiT(DiT(in_channels=dit_in_channels), [DiT(in_channels=dit_in_channels, add_zero_convs=True) for _ in range(num_control_nets)])
        else:
            self.dit = DiT(in_channels=dit_in_channels)

        self.sampling_steps = sampling_steps

        self.schedule = LinearSchedule(T=1000)
        self.sampling_timesteps = Timesteps(
            T=self.schedule.T,
            steps=self.sampling_steps,
        )
        self.sampler = EulerSampler(
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
            prediction_type='velocity'
        )

        # required for ConfigMixin
        self.register_to_config(
            sampling_steps=sampling_steps,
            depth_anything_v2_encoder=depth_anything_v2_encoder,
            depth_anything_v2_features=depth_anything_v2_features,
            depth_anything_v2_out_channels=depth_anything_v2_out_channels,
            dit_in_channels=dit_in_channels,
            num_control_nets=num_control_nets,
        )
    
    @torch.no_grad()
    def infer_image(self, image_bgr_hwc, use_fp16: bool = True):
        # Resize the image to match the training resolution area while keeping the original aspect ratio.
        resize_image_bgr_hwc = resize_keep_aspect(image_bgr_hwc)
        image_rgb_1chw = image2tensor(resize_image_bgr_hwc)
        image_rgb_1chw = image_rgb_1chw.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            depth = self.forward_test(image_rgb_1chw)
        return depth, resize_image_bgr_hwc
    
    @torch.no_grad()
    def forward_test(self, image_rgb_1chw):

        if self.config.num_control_nets > 0:
            raise NotImplementedError("ControlNet-style PPD not implemented yet for forward_test")

        semantics = self.semantics_prompt(image_rgb_1chw)
        cond = image_rgb_1chw - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(self.device)

        depth_sequence = []

        depth_sequence.append(latent.cpu().numpy())
        
        for timestep in self.sampling_timesteps:
            input = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=input, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)
            depth_sequence.append(latent.cpu().numpy())
        
        if os.environ.get("EXPORT_GIF", "0") == "1":
            
            final_depth = depth_sequence[-1]
            vmin = float(final_depth.min())
            vmax = float(final_depth.max())
            
            def colorize_internal(value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"):
                colored = colorize_depth_maps(value.squeeze(0), vmin, vmax, cmap)
                colored = (colored * 255).astype(np.uint8)
                colored_hwc = chw2hwc(colored.squeeze(0))
                return Image.fromarray(colored_hwc)
            
            # let's interpolate between them!
            num_frames = len(depth_sequence)
            interpolated_frames = []
            for i in range(num_frames - 1):
                prev_frame = depth_sequence[i]
                next_frame = depth_sequence[i + 1]
                interpolation_levels = torch.linspace(0, 1, 5)
                interpolated_frames.append(prev_frame)
                for t in interpolation_levels:
                    interpolated_frame = prev_frame * (1 - t.item()) + next_frame * t.item()
                    interpolated_frames.append(interpolated_frame)
            for i in range(10):
                interpolated_frames.append(next_frame)
                
            # let's save it as a gif!
            images = [colorize_internal(frame, vmin=vmin, vmax=vmax) for frame in interpolated_frames]
            images[0].save("/tmp/depth_sequence.gif", save_all=True, append_images=images[1:], duration=10, loop=0)

            
            # # Save each step as /tmp/0_depth.jpg, /tmp/1_depth.jpg, etc.
            # for i, depth in enumerate(depth_sequence):
            #     img = colorize_internal(depth, vmin=vmin, vmax=vmax)
            #     img.save(f"/tmp/{i}_depth.jpg")

        return latent + 0.5


    @torch.no_grad()
    def semantics_prompt(self, image_rgb_hwc):
        with torch.no_grad():
            semantics = self.semantics_encoder(image_rgb_hwc)
        return semantics
    
    # we define forward() to just be .dit() since .dit is the only trainable part of the network
    def forward(self, *args, **kwargs):
        return self.dit(*args, **kwargs)
    
    def requires_grad_(self, value):
        self.semantics_encoder.requires_grad_(False)
        self.dit.requires_grad_(value)
        return self