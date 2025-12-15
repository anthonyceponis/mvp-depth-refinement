#export WORKSPACE_DIR="$(dirname $0)/../.."
export WORKSPACE_DIR=$BASE_DATA_DIR../
export PYTHONPATH="$WORKSPACE_DIR":$PYTHONPATH

echo $WORKSPACE_DIR
echo $PYTHONPATH

num_gpus=$NUM_GPUS
macrobatch_size=1

gradient_accumulation_steps=$((macrobatch_size / num_gpus))

# to resume:
# --learning_rate=1e-6 --student_ckpt_dir_revision reverse-simple-transformation
    # --student_ckpt_dir_revision reverse-simple-transformation \
accelerate launch --num_processes $num_gpus ppd_sharpdepth/training/train.py \
    --sds_loss_weight 0.1 \
    --depth_weight 0.4 \
    --base_ckpt_dir andrew-healey/sharpdepth \
    --student_ckpt_dir andrew-healey/sharpdepth \
    --student_ckpt_dir_revision trained/edge_loss_unidepth/checkpoint-2000 \
    --add_datetime_prefix \
    --report_to wandb \
    --mixed_precision bf16 \
    --seed 42 \
    --allow_tf32 \
    --learning_rate 1e-6 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --tracker_project_name ppd_sharpdepth_controlnet_train \
    --set_grads_to_none \
    --checkpointing_steps 1000 \
    --validation_steps 200 \
    --train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 1 \
    --use_ema \
    --base_data_dir "$WORKSPACE_DIR/data/" \
    --config "$WORKSPACE_DIR/config/train_marigold_depth.yaml" \
    --output_dir "$WORKSPACE_DIR/train_output_edge_loss_unidepth_1/" \
    --base_model unidepth \
    --denoiser pixel_perfect_depth_controlnet \
    --use_conditioning_probability 0.8 \
    --dit_patch_encoder_lr_multiplier 1 \
    --blur_unidepth_output_ratio 64 \
    --noise_aware_latent_noise_scale 0.0 \
    --gradient_checkpointing \
    --log_depth_maps \
    --depth_loss_away_from_edges_threshold_px 16 \
    --edge_loss_blur_radius_px 8 \
    --forward_diffuse_from initial_pred_depth \
    --forward_diffuse_from_initial_pred_depth_probability 0.25 \
    --use_synthetic_conditioning_probability 0.25 \
    --use_edge_loss_as_sds_loss \
    --wandb_name "controlnet_edge_loss_unidepth" \
    "$@"