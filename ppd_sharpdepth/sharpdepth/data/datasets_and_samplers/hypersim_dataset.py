# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from ppd_sharpdepth.sharpdepth.data.datasets_and_samplers.base_depth_dataset import BaseDepthDataset, DepthFileNameMode, get_pred_name, DatasetMode
import numpy as np
import os
from PIL import Image
import pandas as pd

class HypersimDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # NOTE: Copied from marigold. 
            min_depth=1e-5,
            max_depth=65.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_i_d,
            **kwargs,
        )

        BASE_DATA_DIR = os.environ["BASE_DATA_DIR"]
        intrinsics_filepath = f"{BASE_DATA_DIR}/../data_split/hypersim_normals/metadata_camera_parameters.csv"

        self.intrinsics_df = pd.read_csv(intrinsics_filepath).set_index("scene_name")


    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        return rgb_data

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}
        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        scene = rgb_rel_path.split("/")[0]
        scene_intrisincs = self.intrinsics_df.loc[scene]
        fx, fy, cx, cy = (
            scene_intrisincs["M_proj_00"],
            scene_intrisincs["M_proj_11"],
            scene_intrisincs["M_proj_02"],
            scene_intrisincs["M_proj_12"],
        )
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path
            )
            rasters.update(depth_data)
 
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()

        other = {
            "index": index,
            "rgb_relative_path": rgb_rel_path,
            "disp_name": self.disp_name,
            "intrinsics": intrinsics,
        }
 
        return rasters, other
