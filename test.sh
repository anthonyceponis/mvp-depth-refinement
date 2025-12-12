for base_model in  "unidepth" "pixel_perfect_depth" "depth_anything_small"; do 
  python -m ppd_sharpdepth.infer --checkpoint submodules/SharpDepth/checkpoints/sharpdepth --output_dir /tmp/sharpdepth_out_viz/ --input_dir submodules/SharpDepth/assets/in-the-wild_example --base_model $base_model
done
