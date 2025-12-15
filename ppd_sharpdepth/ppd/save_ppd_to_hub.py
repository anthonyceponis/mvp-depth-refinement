from huggingface_hub import create_branch, delete_folder, upload_folder

from ppd_sharpdepth.ppd.models.dit import ControlNetDiT, DiT


if __name__ == "__main__":
    import sys
    print(f"sys.argv: {sys.argv}")

    from huggingface_hub import hf_hub_download
    import torch
    from ppd_sharpdepth.ppd.models.ppd import PixelPerfectDepth

    device = torch.device('cuda')

    if sys.argv[1] == "push":
        print("Pushing Pixel Perfect Depth to Hugging Face Hub")

        ckpt_path = hf_hub_download(repo_id="gangweix/pixel-perfect-depth", filename="ppd.pth")
        semantics_path = hf_hub_download(repo_id="depth-anything/Depth-Anything-V2-Large", filename="depth_anything_v2_vitl.pth")

        base_config = {
            "sampling_steps": 4,
            "depth_anything_v2_encoder": "vitl",
            "depth_anything_v2_features": 256,
            "depth_anything_v2_out_channels": [256, 512, 1024, 1024],
        }

        model = PixelPerfectDepth(semantics_pth=semantics_path, **base_config, dit_in_channels=4)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model = model.to(device).eval()
        model.requires_grad_(False)

        model.push_to_hub("andrew-healey/sharpdepth", subfolder="ppd")

        state_dict = model.state_dict()
        old_weight = state_dict["dit.x_embedder.proj.weight"]
        N, C, H, W = old_weight.shape
        assert C == 4
        with torch.no_grad():
            new_weight = torch.zeros(size=[N, 5, H, W])
            new_weight[:, :4, :, :] = old_weight
            new_weight[:, 4, :, :] = 0
        state_dict["dit.x_embedder.proj.weight"] = new_weight

        multi_channel_model = PixelPerfectDepth(**base_config, dit_in_channels=5)
        multi_channel_model.load_state_dict(state_dict)

        multi_channel_model.push_to_hub("andrew-healey/sharpdepth", subfolder="ppd_student")

    elif sys.argv[1] == "pull":
        print("Pulling Pixel Perfect Depth from Hugging Face Hub")
        model = PixelPerfectDepth.from_pretrained("andrew-healey/sharpdepth", subfolder="ppd")
        model = PixelPerfectDepth.from_pretrained("andrew-healey/pixel-perfect-depth")
        print("Pulled!")
    elif sys.argv[1] == "push_trained_checkpoint":
        branch_name = sys.argv[2]
        local_folder_name = sys.argv[3]
        print(f"Pushing trained checkpoint in local folder {local_folder_name} to remote ppd_student/ subfolder in a new branch {branch_name} of andrew-healey/sharpdepth")
        input("Press Enter to continue: ")

        branch = create_branch(repo_id="andrew-healey/sharpdepth", branch=branch_name,revision="main")
        # rm -rf ppd_student on the branch
        delete_folder(repo_id="andrew-healey/sharpdepth", path_in_repo="ppd_student", revision=branch_name)
        upload_folder(repo_id="andrew-healey/sharpdepth", folder_path=local_folder_name, path_in_repo="ppd_student", revision=branch_name)
    elif sys.argv[1] == "push_trained_checkpoint_controlnet":
        branch_name = sys.argv[2]
        local_folder_name = sys.argv[3]
        print(f"Pushing trained checkpoint in local folder {local_folder_name} to remote ppd_student_controlnet/ subfolder in a new branch {branch_name} of andrew-healey/sharpdepth")
        input("Press Enter to continue: ")

        try:
            branch = create_branch(repo_id="andrew-healey/sharpdepth", branch=branch_name,revision="main")
        except:
            pass
        try:
            # rm -rf ppd_student on the branch
            delete_folder(repo_id="andrew-healey/sharpdepth", path_in_repo="ppd_student_controlnet", revision=branch_name)
        except:
            pass
        upload_folder(repo_id="andrew-healey/sharpdepth", folder_path=local_folder_name, path_in_repo="ppd_student_controlnet", revision=branch_name)
    elif sys.argv[1] == "push_trained_checkpoint_controlnet":
        branch_name = sys.argv[2]
        local_folder_name = sys.argv[3]
        print(f"Pushing trained checkpoint in local folder {local_folder_name} to remote ppd_student_controlnet/ subfolder in a new branch {branch_name} of andrew-healey/sharpdepth")
        input("Press Enter to continue: ")

        branch = create_branch(repo_id="andrew-healey/sharpdepth", branch=branch_name,revision="main")
        # rm -rf ppd_student on the branch
        delete_folder(repo_id="andrew-healey/sharpdepth", path_in_repo="ppd_student_controlnet", revision=branch_name)
        upload_folder(repo_id="andrew-healey/sharpdepth", folder_path=local_folder_name, path_in_repo="ppd_student_controlnet", revision=branch_name)
    elif sys.argv[1] == "push_lotus_student":
        print("Pushing Lotus Student to Hugging Face Hub")
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained("andrew-healey/sharpdepth", subfolder="unet")
        model.push_to_hub("andrew-healey/sharpdepth", subfolder="unet_student")
    elif sys.argv[1] == "push_controlnet":
        # let's make a controlnet with two forms of conditioning! one for the ppd depth map, one for the unidepth depth map.

        print("Pushing Pixel Perfect Depth to Hugging Face Hub")

        ckpt_path = hf_hub_download(repo_id="gangweix/pixel-perfect-depth", filename="ppd.pth")
        semantics_path = hf_hub_download(repo_id="depth-anything/Depth-Anything-V2-Large", filename="depth_anything_v2_vitl.pth")

        base_config = {
            "sampling_steps": 4,
            "depth_anything_v2_encoder": "vitl",
            "depth_anything_v2_features": 256,
            "depth_anything_v2_out_channels": [256, 512, 1024, 1024],
            "dit_in_channels": 4,
        }

        model = PixelPerfectDepth(semantics_pth=semantics_path, **base_config)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model = model.to(device).eval()
        model.requires_grad_(False)

        base_dit = model.dit
        assert base_dit.add_zero_convs == False
        cond_dit_config = {
            "in_channels": base_dit.in_channels,
            "out_channels": base_dit.out_channels,
            "hidden_size": 1024,
            "depth": len(base_dit.blocks),
            "num_heads": base_dit.num_heads,
            "mlp_ratio": 4.0,
            "add_zero_convs": True,
        }
        conditioning_dits = [DiT(**cond_dit_config) for _ in range(2)]
        for conditioning_dit in conditioning_dits:
            conditioning_dit.load_state_dict(base_dit.state_dict(), strict=False)

        controlnet_model = ControlNetDiT(base_dit, conditioning_dits)

        modified_ppd_model = PixelPerfectDepth(**base_config, num_control_nets=2)
        modified_ppd_model.load_state_dict(model.state_dict(), strict=False)
        modified_ppd_model.dit = controlnet_model
        modified_ppd_model.push_to_hub("andrew-healey/sharpdepth", subfolder="ppd_student_controlnet")

    else:
        raise ValueError(f"Invalid command: {sys.argv[1]}")